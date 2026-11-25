//! A tiny one-shot signal for the browser build, where nothing can block: wgpu completion
//! callbacks (`map_async`, `on_submitted_work_done`) fire from the JS event loop, and the future
//! that awaits them must also notice a lost device, whose callback can arrive instead.

// Only the browser build uses it; native builds compile it for the tests below.
#![cfg_attr(not(target_arch = "wasm32"), allow(dead_code))]

use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll, Waker};

struct State<T> {
    value: Option<T>,
    waker: Option<Waker>,
}

/// The sending half: call [`Sender::send`] from a wgpu callback.
pub struct Sender<T>(Arc<Mutex<State<T>>>);

/// The receiving half: a future that resolves with the sent value.
pub struct Receiver<T>(Arc<Mutex<State<T>>>);

pub fn channel<T>() -> (Sender<T>, Receiver<T>) {
    let state = Arc::new(Mutex::new(State {
        value: None,
        waker: None,
    }));
    (Sender(state.clone()), Receiver(state))
}

impl<T> Sender<T> {
    pub fn send(self, value: T) {
        let waker = {
            let mut s = self.0.lock().unwrap_or_else(|e| e.into_inner());
            s.value = Some(value);
            s.waker.take()
        };
        if let Some(w) = waker {
            w.wake();
        }
    }
}

impl<T> Receiver<T> {
    /// Resolves with the sent value, or with `Err(reason)` as soon as `abort` returns one (it is
    /// called on every poll with the task's waker, so the caller can register it to be woken when
    /// the answer may change, for example from the device-lost callback).
    pub fn recv_or(
        self,
        abort: impl Fn(&Waker) -> Option<String> + Unpin,
    ) -> impl Future<Output = Result<T, String>> {
        Recv { rx: self, abort }
    }
}

struct Recv<T, F> {
    rx: Receiver<T>,
    abort: F,
}

impl<T, F: Fn(&Waker) -> Option<String> + Unpin> Future for Recv<T, F> {
    type Output = Result<T, String>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let this = self.get_mut();
        let mut s = this.rx.0.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(v) = s.value.take() {
            return Poll::Ready(Ok(v));
        }
        if let Some(why) = (this.abort)(cx.waker()) {
            return Poll::Ready(Err(why));
        }
        s.waker = Some(cx.waker().clone());
        Poll::Pending
    }
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn a_value_sent_before_the_await_resolves_ok() {
        let (tx, rx) = channel();
        tx.send(7u32);
        assert_eq!(pollster::block_on(rx.recv_or(|_| None)), Ok(7));
    }

    #[test]
    fn a_value_sent_from_another_thread_wakes_the_pending_future() {
        let (tx, rx) = channel();
        let sender = std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(50));
            tx.send("done");
        });
        assert_eq!(pollster::block_on(rx.recv_or(|_| None)), Ok("done"));
        sender.join().unwrap();
    }

    #[test]
    fn an_abort_reason_resolves_the_future_with_an_error() {
        let (_tx, rx) = channel::<u32>();
        assert_eq!(
            pollster::block_on(rx.recv_or(|_| Some("device lost".to_string()))),
            Err("device lost".to_string())
        );
    }

    #[test]
    fn a_sent_value_wins_over_an_abort_reason() {
        let (tx, rx) = channel();
        tx.send(1u8);
        assert_eq!(
            pollster::block_on(rx.recv_or(|_| Some("late".to_string()))),
            Ok(1)
        );
    }

    #[test]
    fn the_abort_check_sees_the_waker_so_a_loss_can_wake_the_task() {
        use std::sync::atomic::{AtomicBool, Ordering};
        let (_tx, rx) = channel::<u32>();
        let lost = Arc::new(AtomicBool::new(false));
        let waker_slot: Arc<Mutex<Option<Waker>>> = Arc::new(Mutex::new(None));
        let (l, w) = (lost.clone(), waker_slot.clone());
        let flipper = std::thread::spawn(move || {
            // Wait until the future has registered its waker, then "lose the device" and wake it.
            loop {
                if let Some(waker) = w.lock().unwrap().clone() {
                    l.store(true, Ordering::SeqCst);
                    waker.wake();
                    return;
                }
                std::thread::sleep(Duration::from_millis(5));
            }
        });
        let out = pollster::block_on(rx.recv_or(move |waker| {
            *waker_slot.lock().unwrap() = Some(waker.clone());
            lost.load(Ordering::SeqCst)
                .then(|| "device lost".to_string())
        }));
        assert_eq!(out, Err("device lost".to_string()));
        flipper.join().unwrap();
    }
}
