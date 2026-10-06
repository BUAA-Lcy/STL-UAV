# NewRealDemo execution scope

The user explicitly reset this effort to a real-video demo from scratch.
For this directory, GPS and camera/gimbal metadata are allowed. Do not import
legacy retrieval, VOP, visual matching, rescue heuristics or legacy result files.
Use original video and raw satellite imagery only. Preserve the rest of the repo.

The current requested deliverable is the last 30 seconds of the Aerial-to-ground2
185426 recording. Default CLI behavior must retain its source frame indices and
frame rate. See ../../NewRealDemo.md for the data contract and commands.

Do not label GPS-assisted correspondence as visual localization or pixel-level
registration. Future algorithm work requires a separate discussion with the user.
The user subsequently explicitly requested high-resolution tile downloads after
being informed that Esri/ArcGIS requests disclose the route location. Requests
to Esri imagery services for this recording are now authorized. Cache and validate
real imagery; never present missing-image placeholders or upsampling as HD data.

The user also chose the Google Earth export workflow for this route. Prepare and
georeference actual Earth exports with artificial KML control markers. Preserve
the original exports and Google/provider attribution. Export calibration is not
UAV-to-satellite visual matching. Do not select the new imagery before coverage,
control-point checks and the clean-export check pass.
