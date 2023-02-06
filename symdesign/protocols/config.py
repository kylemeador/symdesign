from symdesign import flags


run_on_pose_job = (
    flags.orient,
    flags.expand_asu,
    flags.rename_chains,
    flags.check_clashes,
    flags.generate_fragments,
    flags.interface_metrics,
    flags.optimize_designs,
    flags.refine,
    flags.interface_design,
    flags.design,
    flags.analysis,
    flags.predict_structure,
    flags.process_rosetta_metrics,
    flags.nanohedra
)
returns_pose_jobs = (
    flags.nanohedra,
    flags.select_poses,
    flags.select_designs,
    flags.select_sequences
)
