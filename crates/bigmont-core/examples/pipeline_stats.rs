//! Compile one kernel configuration through the real Vulkan driver and print what the driver
//! reports about it (VK_KHR_pipeline_executable_properties): registers, LDS, scratch (spills),
//! instruction counts, and save any text internal representation (for example AMD ISA).
//!
//!   cargo run --release -p bigmont-core --example pipeline_stats -- TPI WG [DEVICE_NAME_SUBSTRING [rolled|unrolled [wgpu|flags|plain [fused|symmetric|symmetric-pairs]]]]
//!
//! Mode `wgpu` (default) mirrors what wgpu-hal 30.0.2 does for a compute pipeline on a device that
//! has Features::SUBGROUP: VK_EXT_subgroup_size_control (Vulkan 1.3 feature) is enabled and the stage
//! is created with PipelineShaderStageCreateFlags::ALLOW_VARYING_SUBGROUP_SIZE. wgpu sets no
//! required subgroup size and not REQUIRE_FULL_SUBGROUPS (see wgpu-hal src/vulkan/device.rs,
//! `compile_stage`) and the SPIR-V is written with the same naga options as wgpu-hal builds
//! (SPIR-V 1.6, bounds-check policy Restrict, native workgroup zero-init, group-non-uniform
//! capabilities). Mode `plain` creates the stage with no flags and no feature and uses naga's default
//! options (the earlier behaviour of this tool), for comparison. Mode `flags` sets only the stage
//! flag and feature, with naga's default options, to separate the two effects.
//!
//! Adapted from the equivalent diagnostic in the gapsieve project (same author, MIT OR Apache-2.0).
//! It talks to Vulkan directly with `ash`, so it shows what the driver does with the SPIR-V that
//! naga generates; wgpu may add options of its own. Only drivers that implement
//! VK_KHR_pipeline_executable_properties report anything (AMD does; NVIDIA's proprietary driver
//! generally does not).

use std::ffi::{CStr, CString};

use ash::vk;
use bigmont_core::kernel::{KernelConfig, Variant, parse_squaring};

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let tpi: u32 = args
        .first()
        .and_then(|v| v.parse().ok())
        .expect("usage: pipeline_stats TPI WG [DEVICE_SUBSTRING]");
    let wg: u32 = args
        .get(1)
        .and_then(|v| v.parse().ok())
        .expect("usage: pipeline_stats TPI WG [DEVICE_SUBSTRING]");
    let want = args.get(2).map(|s| s.to_lowercase());
    let variant = match args.get(3).map(String::as_str) {
        None | Some("rolled") => Variant::Rolled,
        Some("unrolled") => Variant::Unrolled,
        Some(other) => panic!("fourth argument must be rolled or unrolled, got {other}"),
    };
    // wgpu: stage flag + wgpu's naga SPIR-V options; flags: only the stage flag (naga defaults); plain: neither.
    let (stage_flag_on, wgpu_mode, mode_name) = match args.get(4).map(String::as_str) {
        None | Some("wgpu") => (true, true, "wgpu"),
        Some("flags") => (true, false, "flags"),
        Some("plain") => (false, false, "plain"),
        Some(other) => panic!("fifth argument must be wgpu, flags or plain, got {other}"),
    };
    let name = args.get(5).map_or("fused", String::as_str);
    let (squaring, packing) = parse_squaring(name).unwrap_or_else(|| {
        panic!("sixth argument must be fused, symmetric or symmetric-pairs, got {name}")
    });
    let cfg = KernelConfig::new(tpi, wg, variant)
        .with_squaring(squaring)
        .with_packing(packing);
    if let Err(e) = cfg.validate() {
        // The unrolled variant has no tpi 1 kernel: say so and stop (not an error).
        println!("skipped: {e}");
        return;
    }
    let out_dir = std::env::temp_dir().join("bigmont-isa");
    std::fs::create_dir_all(&out_dir).unwrap();

    let source = cfg.wgsl();
    let module = naga::front::wgsl::parse_str(&source).expect("WGSL parse");
    let mut caps = naga::valid::Capabilities::empty();
    if cfg.needs_subgroups() {
        caps |= naga::valid::Capabilities::SUBGROUP | naga::valid::Capabilities::SUBGROUP_BARRIER;
    }
    let info = naga::valid::Validator::new(naga::valid::ValidationFlags::all(), caps)
        .validate(&module)
        .expect("WGSL validation");
    let name = "main".to_string();

    unsafe {
        let entry = ash::Entry::load().expect("no Vulkan loader");
        let app = vk::ApplicationInfo::default().api_version(vk::API_VERSION_1_3);
        let instance = entry
            .create_instance(
                &vk::InstanceCreateInfo::default().application_info(&app),
                None,
            )
            .unwrap();
        let devices = instance.enumerate_physical_devices().unwrap();
        let pdev = devices
            .into_iter()
            .find(|&d| {
                let p = instance.get_physical_device_properties(d);
                let n = CStr::from_ptr(p.device_name.as_ptr())
                    .to_string_lossy()
                    .to_lowercase();
                want.as_ref().is_none_or(|w| n.contains(w))
            })
            .expect("no matching Vulkan device");
        let props = instance.get_physical_device_properties(pdev);
        println!(
            "device: {}",
            CStr::from_ptr(props.device_name.as_ptr()).to_string_lossy()
        );
        println!(
            "config: tpi {tpi}, wg {wg}, limbs per lane {}, variant {variant:?}, mode {mode_name}",
            cfg.limbs_per_lane()
        );
        let family = instance
            .get_physical_device_queue_family_properties(pdev)
            .iter()
            .position(|q| q.queue_flags.contains(vk::QueueFlags::COMPUTE))
            .unwrap() as u32;
        let priorities = [1.0f32];
        let queues = [vk::DeviceQueueCreateInfo::default()
            .queue_family_index(family)
            .queue_priorities(&priorities)];
        let extensions = [ash::khr::pipeline_executable_properties::NAME.as_ptr()];
        let mut exec_props = vk::PhysicalDevicePipelineExecutablePropertiesFeaturesKHR::default()
            .pipeline_executable_info(true);
        let mut vk13 = vk::PhysicalDeviceVulkan13Features::default()
            .shader_zero_initialize_workgroup_memory(true)
            .subgroup_size_control(stage_flag_on);
        let device = instance
            .create_device(
                pdev,
                &vk::DeviceCreateInfo::default().queue_create_infos(&queues).enabled_extension_names(&extensions).push_next(&mut exec_props).push_next(&mut vk13),
                None,
            )
            .expect("device with VK_KHR_pipeline_executable_properties (this driver may not support it)");
        let pep = ash::khr::pipeline_executable_properties::Device::new(&instance, &device);

        let (index, _) = module
            .entry_points
            .iter()
            .enumerate()
            .find(|(_, e)| e.name == name)
            .expect("no entry point main");
        let usage = info.get_entry_point(index);
        let bindings: Vec<_> = module
            .global_variables
            .iter()
            .filter_map(|(handle, var)| {
                let b = var.binding.as_ref()?;
                if usage[handle].is_empty() {
                    return None;
                }
                let ty = match var.space {
                    naga::AddressSpace::Uniform => vk::DescriptorType::UNIFORM_BUFFER,
                    naga::AddressSpace::Storage { .. } => vk::DescriptorType::STORAGE_BUFFER,
                    other => panic!("unexpected address space {other:?}"),
                };
                Some(
                    vk::DescriptorSetLayoutBinding::default()
                        .binding(b.binding)
                        .descriptor_type(ty)
                        .descriptor_count(1)
                        .stage_flags(vk::ShaderStageFlags::COMPUTE),
                )
            })
            .collect();
        let set_layout = device
            .create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings),
                None,
            )
            .unwrap();
        let set_layouts = [set_layout];
        let layout = device
            .create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default().set_layouts(&set_layouts),
                None,
            )
            .unwrap();
        let options = naga::back::spv::PipelineOptions {
            shader_stage: naga::ShaderStage::Compute,
            entry_point: name.clone(),
        };
        let spv_options = if wgpu_mode {
            // Mirrors wgpu-hal 30.0.2 `naga_options` (src/vulkan/adapter.rs) for a Vulkan 1.3 device with
            // Features::SUBGROUP, default runtime checks (bounds checks and loop bounding on),
            // native zero-initialisation of workgroup memory, and robustBufferAccess2 absent (so
            // storage-buffer accesses are clamped by naga with BoundsCheckPolicy::Restrict).
            use naga::back::spv::Capability as C;
            let capabilities = [
                C::Shader,
                C::Matrix,
                C::Sampled1D,
                C::Image1D,
                C::ImageQuery,
                C::DerivativeControl,
                C::StorageImageExtendedFormats,
                C::GroupNonUniform,
                C::GroupNonUniformVote,
                C::GroupNonUniformArithmetic,
                C::GroupNonUniformBallot,
                C::GroupNonUniformShuffle,
                C::GroupNonUniformShuffleRelative,
                C::GroupNonUniformQuad,
            ];
            naga::back::spv::Options {
                lang_version: (1, 6),
                flags: naga::back::spv::WriterFlags::FORCE_POINT_SIZE,
                capabilities: Some(capabilities.iter().cloned().collect()),
                bounds_check_policies: naga::proc::BoundsCheckPolicies {
                    index: naga::proc::BoundsCheckPolicy::Restrict,
                    buffer: naga::proc::BoundsCheckPolicy::Restrict,
                    image_load: naga::proc::BoundsCheckPolicy::Restrict,
                    binding_array: naga::proc::BoundsCheckPolicy::Unchecked,
                },
                zero_initialize_workgroup_memory:
                    naga::back::spv::ZeroInitializeWorkgroupMemoryMode::Native,
                force_loop_bounding: true,
                fake_missing_bindings: false,
                // wgpu builds this from the bind group layout; for these four bindings it is the identity map on set 0.
                binding_map: (0..4)
                    .map(|b| {
                        (
                            naga::ResourceBinding {
                                group: 0,
                                binding: b,
                            },
                            naga::back::spv::BindingInfo {
                                descriptor_set: 0,
                                binding: b,
                                binding_array_size: None,
                            },
                        )
                    })
                    .collect(),
                ..Default::default()
            }
        } else {
            naga::back::spv::Options::default()
        };
        // wgpu writes the whole module (no entry point filter); this module has the single entry point `main`.
        let spirv = naga::back::spv::write_vec(
            &module,
            &info,
            &spv_options,
            if wgpu_mode { None } else { Some(&options) },
        )
        .expect("SPIR-V");
        println!("spirv: {} words, version {:#x}", spirv.len(), spirv[1]);
        let shader = device
            .create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&spirv), None)
            .unwrap();
        let main = CString::new(name.as_str()).unwrap();
        let stage_flags = if stage_flag_on {
            vk::PipelineShaderStageCreateFlags::ALLOW_VARYING_SUBGROUP_SIZE
        } else {
            vk::PipelineShaderStageCreateFlags::empty()
        };
        let stage = vk::PipelineShaderStageCreateInfo::default()
            .flags(stage_flags)
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(shader)
            .name(&main);
        let create = [vk::ComputePipelineCreateInfo::default()
            .stage(stage)
            .layout(layout)
            .flags(
                vk::PipelineCreateFlags::CAPTURE_STATISTICS_KHR
                    | vk::PipelineCreateFlags::CAPTURE_INTERNAL_REPRESENTATIONS_KHR,
            )];
        let pipeline =
            match device.create_compute_pipelines(vk::PipelineCache::null(), &create, None) {
                Ok(p) => p[0],
                Err((_, e)) => {
                    println!("pipeline creation failed: {e:?}");
                    return;
                }
            };
        let executables = pep
            .get_pipeline_executable_properties(&vk::PipelineInfoKHR::default().pipeline(pipeline))
            .unwrap();
        for (ei, ex) in executables.iter().enumerate() {
            println!(
                "== {}: {} (subgroup size {})",
                name,
                CStr::from_ptr(ex.name.as_ptr()).to_string_lossy(),
                ex.subgroup_size
            );
            let exec = vk::PipelineExecutableInfoKHR::default()
                .pipeline(pipeline)
                .executable_index(ei as u32);
            for st in pep.get_pipeline_executable_statistics(&exec).unwrap() {
                let value = match st.format {
                    vk::PipelineExecutableStatisticFormatKHR::BOOL32 => {
                        (st.value.b32 != 0).to_string()
                    }
                    vk::PipelineExecutableStatisticFormatKHR::INT64 => st.value.i64.to_string(),
                    vk::PipelineExecutableStatisticFormatKHR::UINT64 => st.value.u64.to_string(),
                    vk::PipelineExecutableStatisticFormatKHR::FLOAT64 => st.value.f64.to_string(),
                    _ => "?".into(),
                };
                println!(
                    "   {:<32} {value}",
                    CStr::from_ptr(st.name.as_ptr()).to_string_lossy()
                );
            }
            let get = pep
                .fp()
                .get_pipeline_executable_internal_representations_khr;
            let mut count = 0u32;
            let _ = get(device.handle(), &exec, &mut count, std::ptr::null_mut());
            let mut reps =
                vec![vk::PipelineExecutableInternalRepresentationKHR::default(); count as usize];
            let _ = get(device.handle(), &exec, &mut count, reps.as_mut_ptr());
            let mut bufs: Vec<Vec<u8>> = reps.iter().map(|r| vec![0u8; r.data_size]).collect();
            for (r, b) in reps.iter_mut().zip(bufs.iter_mut()) {
                r.p_data = b.as_mut_ptr().cast();
            }
            let _ = get(device.handle(), &exec, &mut count, reps.as_mut_ptr());
            for (r, b) in reps.iter().zip(&bufs) {
                let rep = CStr::from_ptr(r.name.as_ptr())
                    .to_string_lossy()
                    .replace([' ', '/'], "_");
                let path = out_dir.join(format!(
                    "{}_tpi{tpi}_wg{wg}_{mode_name}_{rep}.{}",
                    format!("{variant:?}").to_lowercase(),
                    if r.is_text != 0 { "txt" } else { "bin" }
                ));
                std::fs::write(&path, b.split(|&c| c == 0).next().unwrap_or(&[])).unwrap();
                println!("   saved {} ({} bytes)", path.display(), r.data_size);
            }
        }
    }
}
