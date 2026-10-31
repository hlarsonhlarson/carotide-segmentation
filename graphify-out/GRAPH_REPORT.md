# Graph Report - carotid-segmentation  (2026-10-23)

## Corpus Check
- 57 files · ~158,498 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 2780 file(s) not represented in the graph (top: .pcm 270, .swiftmodule 48, .d 25)

## Summary
- 1026 nodes · 2283 edges · 66 communities (59 shown, 7 thin omitted)
- Extraction: 94% EXTRACTED · 6% INFERRED · 0% AMBIGUOUS · INFERRED: 141 edges (avg confidence: 0.85)
- Token cost: 68,488 input · 0 output

## Community Hubs (Navigation)
- Swift APIClient (backend HTTP calls)
- Boundary Conditions UI + SU2 params state
- TotalSegmentator bootstrap steps
- AppState + BoundaryConditionsView core
- Interactive 3D plane/scene views (AppKit)
- build_surface.py (contour-to-mesh rasterization)
- carotid_seg meshing (surface build/clip/clean)
- carotid_seg pipeline orchestration
- FastAPI server app module
- Centerline points panel UI
- Plane pane view (slice rendering UI)
- README overview + Swift build target
- cca_waveform.py (pulsatile inlet waveform)
- AppState centerline/slice navigation
- path_series.py (arc-length resampling)
- carotid_seg treeanalysis (centerline graph)
- SurfacePreviewView (STL parsing/rendering)
- APIClient CodingKeys (JSON field mapping)
- Scene3DGeometry (SceneKit mesh building)
- su2_run.py CFD run endpoints
- AppState project load/save actions
- server app endpoints (autoseed/build/open)
- PathPreview3DView (SceneKit path preview)
- Interactive3DSceneView orbit camera
- AppState SU2 mesh/run triggers + SliceView workflow
- server app contour/polygon geometry helpers
- server app centerline point endpoints
- SegmentPreview3DView (segmentation 3D preview)
- carotid_seg branchtrim + centerlines
- AppState PlaneAxis enum
- Scene3DController (SceneKit lifecycle)
- server centerline_path.py (cost-field tracing)
- Swift build graph: debug exe/module nodes
- Swift build graph: plugin-tools exe/module nodes
- server cases.py (CaseState) + DICOM load
- server app case creation/project endpoints
- carotid_seg mra_pipeline (tree extraction)
- Swift build graph: debug target client
- Swift build graph: debug exe build command
- Swift build graph: debug exe entitlements
- Swift build graph: debug module command
- Swift build graph: debug PackageStructure
- Swift build graph: debug swift-version node
- Swift build graph: debug file nodes
- Swift build graph: debug targets
- Swift build graph: plugin-tools target client
- Swift build graph: plugin-tools exe command
- Swift build graph: plugin-tools entitlements
- Swift build graph: plugin-tools module command
- Swift build graph: plugin-tools PackageStructure
- Swift build graph: plugin-tools swift-version node
- Swift build graph: plugin-tools file nodes
- Swift build graph: plugin-tools targets
- README anatomy/TotalSegmentator task terms
- Swift build graph: debug build-sources command
- Swift build graph: debug entitlement-plist command
- Swift build graph: debug LinkFileList command
- Swift build graph: plugin-tools build-sources command
- Swift build graph: plugin-tools entitlement-plist command
- Swift build graph: plugin-tools LinkFileList command
- server app SU2 mesh job orchestration
- server reslice.py (oblique slice sampling)
- carotid_seg mra_pipeline TOF loading
- CarotidSegApp Package.swift manifest
- README ECA term

## God Nodes (most connected - your core abstractions)
1. `AppState` - 80 edges
2. `APIClient` - 53 edges
3. `APIError` - 37 edges
4. `get()` - 35 edges
5. `CodingKeys` - 25 edges
6. `PlaneAxis` - 25 edges
7. `PlanePaneView` - 23 edges
8. `build_side_surface()` - 23 edges
9. `build_surface()` - 21 edges
10. `SegmentPathView` - 20 edges

## Surprising Connections (you probably didn't know these)
- `physicalImageSize()` --calls--> `spacingMM`  [INFERRED]
  CarotidSegApp/Sources/CarotidSegApp/PlanePaneView.swift → CarotidSegApp/Sources/CarotidSegApp/APIClient.swift
- `build_surface()` --calls--> `_resample_isotropic()`  [EXTRACTED]
  server/build_surface.py → carotid_seg/meshing.py
- `build_surface()` --calls--> `_numpy_to_vtk_image()`  [EXTRACTED]
  server/build_surface.py → carotid_seg/meshing.py
- `build_surface()` --calls--> `_largest_component()`  [EXTRACTED]
  server/build_surface.py → carotid_seg/meshing.py
- `build_surface()` --calls--> `count_boundary_loops()`  [EXTRACTED]
  server/build_surface.py → carotid_seg/meshing.py

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **ECA Reconstruction via TotalSegmentator Seeding + Frangi Vesselness + Bifurcation Labeling** — readme_totalsegmentator, readme_frangi_vesselness, readme_bifurcation_detection, readme_eca [EXTRACTED 1.00]
- **carotid_seg Processing Pipeline Stages (vesselness -> treeanalysis -> volumize -> pipeline orchestration)** — readme_carotid_seg_vesselness, readme_carotid_seg_treeanalysis, readme_carotid_seg_volumize, readme_carotid_seg_pipeline_py [EXTRACTED 1.00]

## Communities (66 total, 7 thin omitted)

### Community 0 - "Swift APIClient (backend HTTP calls)"
Cohesion: 0.09
Nodes (23): APIClient, APIError, badImage, badResponse, .errorDescription, server, BuildSurfaceResult, CaseInfo (+15 more)

### Community 1 - "Boundary Conditions UI + SU2 params state"
Cohesion: 0.06
Nodes (31): BoundaryConditionsView, .body, .flowSplitSection, .pressureSection, .waveformChart, .waveformMeanCms, .waveformSection, .body (+23 more)

### Community 2 - "TotalSegmentator bootstrap steps"
Cohesion: 0.06
Nodes (27): cleanup_work_dir(), run_bootstrap_segmentations(), _group_series(), list_series(), load_series(), _read_image(), _slice_sort_key(), to_nifti_tempfile() (+19 more)

### Community 3 - "AppState + BoundaryConditionsView core"
Cohesion: 0.05
Nodes (17): AppKit, .waveformSamples, AppActivator, CarotidSegApp, .body, ContentView, .body, CaseLoaderView (+9 more)

### Community 4 - "Interactive 3D plane/scene views (AppKit)"
Cohesion: 0.08
Nodes (6): InteractivePlaneSurface, SurfaceView, .isFlipped, Callbacks, PaneEventRouter, Region

### Community 5 - "build_surface.py (contour-to-mesh rasterization)"
Cohesion: 0.10
Nodes (20): _blend_contours_by_angle(), _bridge_gap(), build_surface(), _close_spurious_holes(), _contour_radius_and_offset(), _cut_radius_mm(), _inset_series(), _local_open_end() (+12 more)

### Community 6 - "carotid_seg meshing (surface build/clip/clean)"
Cohesion: 0.14
Nodes (16): _bbox_with_margin(), build_side_surface(), _clean(), _clip_away(), count_boundary_loops(), _cut_plane(), _inset_cut_node(), _largest_component() (+8 more)

### Community 7 - "carotid_seg pipeline orchestration"
Cohesion: 0.12
Nodes (13): _bbox_with_margin(), cap_ica_at_landmark(), _load_canonical(), process_case(), mask_for(), _process_side(), find_bifurcation_and_label(), adaptive_lumen_hu_band() (+5 more)

### Community 8 - "FastAPI server app module"
Cohesion: 0.12
Nodes (19): _default_window(), get_centerlines(), get_control_points(), get_mesh(), get_meta(), get_project_dir(), get_segment_slice(), get_series() (+11 more)

### Community 9 - "Centerline points panel UI"
Cohesion: 0.12
Nodes (8): CenterlinePointsPanel, .body, branchColor(), branchDisplayName(), SegmentPathView, .body, .seriesKey, View

### Community 10 - "Plane pane view (slice rendering UI)"
Cohesion: 0.21
Nodes (7): fitRect(), physicalImageSize(), PlanePaneView, .body, .fetchKey, .otherAxes, .sliceCount

### Community 11 - "README overview + Swift build target"
Cohesion: 0.10
Nodes (19): CarotidSegApp build target (arm64-apple-macosx15.0), Swift 6.1.2 Toolchain (swiftlang-6.1.2.1.2 clang-1700.0.13.5), 'carotid' conda environment (TotalSegmentator, torch, nibabel, scikit-image, scipy, networkx, dicom2nifti, SimpleITK, nnunetv2), Carotid Segmentation Pipeline (overall project), carotid_seg/pipeline.py (per-case orchestration, left/right merge, NIfTI I/O), carotid_seg/totalseg_steps.py (TotalSegmentator task wrappers: CCA, ICA, styloid, C1-C7 landmarks), carotid_seg/treeanalysis.py (skeleton graph, cycle-artifact cleanup, bifurcation search, branch labeling), carotid_seg/vesselness.py (adaptive HU banding + Frangi vessel lumen extraction) (+11 more)

### Community 12 - "cca_waveform.py (pulsatile inlet waveform)"
Cohesion: 0.12
Nodes (10): _build_shape_spline(), inlet_velocity_ms(), mean_velocity_ms(), shape(), main(), marker_area_and_flow(), _progress(), calibrate_two_outlet_rcr() (+2 more)

### Community 13 - "AppState centerline/slice navigation"
Cohesion: 0.18
Nodes (5): AppState, .windowLevel, .windowLevelRange, .windowWidth, .windowWidthRange

### Community 14 - "path_series.py (arc-length resampling)"
Cohesion: 0.13
Nodes (9): build_series(), resample_by_arclength(), smooth_path(), tangents_for(), voxel_path_to_world_mm(), _center_contour(), circle_contour(), levelset_contour() (+1 more)

### Community 15 - "carotid_seg treeanalysis (centerline graph)"
Cohesion: 0.19
Nodes (9): _build_graph(), _clean_graph(), _collapse_dense_hubs(), _collapse_small_cycles(), locate_bifurcation(), locate_bifurcation_from_origin(), _prune_spurs(), resolve_cca_ica_path() (+1 more)

### Community 16 - "SurfacePreviewView (STL parsing/rendering)"
Cohesion: 0.14
Nodes (9): parseBinarySTL(), SurfaceGeometryView, SurfacePreviewCoordinator, SurfacePreviewSource, buildSurface, su2Mesh, SurfacePreviewView, .body (+1 more)

### Community 17 - "APIClient CodingKeys (JSON field mapping)"
Cohesion: 0.10
Nodes (19): CodingKeys, cardiacCycleS, caseId, centerline, defaultWindow, description, ecaMapMmHg, edvCms (+11 more)

### Community 18 - "Scene3DGeometry (SceneKit mesh building)"
Cohesion: 0.29
Nodes (12): bestAlignedRing(), buildWallMesh(), closedLoopNode(), crossSectionRingNode(), lineNode(), mmToScene(), resampleClosedLoopMM(), scn_cross() (+4 more)

### Community 19 - "su2_run.py CFD run endpoints"
Cohesion: 0.16
Nodes (9): cancel_su2_run(), _inlet_direction_xyz(), start_su2_run(), cancel_run(), _cfg_text(), mpi_processes(), prepare_run_files(), run_in_background() (+1 more)

### Community 21 - "server app endpoints (autoseed/build/open)"
Cohesion: 0.16
Nodes (9): autoseed(), build_surface_endpoint(), open_project(), _require_project_dir(), save_centerline_to_project(), save_mesh_to_project(), save_segmentation_to_project(), save_su2_params_to_project() (+1 more)

### Community 22 - "PathPreview3DView (SceneKit path preview)"
Cohesion: 0.21
Nodes (4): originMM, PathPreview3DCoordinator, PathPreview3DView, smoothPathPoints()

### Community 23 - "Interactive3DSceneView orbit camera"
Cohesion: 0.17
Nodes (4): Interactive3DSceneView, .isFlipped, OrbitCameraDragHandler, SceneKit

### Community 24 - "AppState SU2 mesh/run triggers + SliceView workflow"
Cohesion: 0.17
Nodes (6): SliceView, .currentWorkflowStep, .path3DPane, .su2MeshControls, .su2RunControls, .surfaceControls

### Community 25 - "server app contour/polygon geometry helpers"
Cohesion: 0.15
Nodes (9): _contour_compactness(), delete_segment_point(), _get_series_point(), ManualContourRequest, _polygon_radius_mm(), segment_point(), segment_point_manual(), SegmentPointRequest (+1 more)

### Community 26 - "server app centerline point endpoints"
Cohesion: 0.22
Nodes (12): add_centerline_point(), AddPointRequest, _check_side_branch(), clear_centerline(), delete_centerline_point(), get_segmentation_series(), move_centerline_point(), prepare_series() (+4 more)

### Community 28 - "carotid_seg branchtrim + centerlines"
Cohesion: 0.26
Nodes (4): trim_to_single_path(), extract_branch_centerlines(), _ordered_path_points_with_radius(), _vox_to_world()

### Community 29 - "AppState PlaneAxis enum"
Cohesion: 0.18
Nodes (8): PlaneAxis, axial, coronal, .id, .label, sagittal, .shapeIndex, .color

### Community 30 - "Scene3DController (SceneKit lifecycle)"
Cohesion: 0.27
Nodes (4): Scene3DController, Scene3DControlsOverlay, .body, .previewPane

### Community 31 - "server centerline_path.py (cost-field tracing)"
Cohesion: 0.33
Nodes (4): _ct_cost_field(), _mr_cost_field(), stitch(), stitch_full_path()

### Community 32 - "Swift build graph: debug exe/module nodes"
Cohesion: 0.22
Nodes (9): inputs, outputs, tool, inputs, outputs, tool, commands, <CarotidSegApp-arm64-apple-macosx15.0-debug.exe> (+1 more)

### Community 33 - "Swift build graph: plugin-tools exe/module nodes"
Cohesion: 0.22
Nodes (9): inputs, outputs, tool, inputs, outputs, tool, commands, <CarotidSegApp-arm64-apple-macosx15.0-debug.exe> (+1 more)

### Community 35 - "server cases.py (CaseState) + DICOM load"
Cohesion: 0.32
Nodes (3): _load_dicom_case(), CaseState, create()

### Community 36 - "server app case creation/project endpoints"
Cohesion: 0.29
Nodes (5): create_case(), LoadCaseRequest, OpenProjectRequest, set_project_dir(), SetProjectRequest

### Community 37 - "carotid_seg mra_pipeline (tree extraction)"
Cohesion: 0.29
Nodes (3): _elongated_components(), extract_carotid_trees(), _regrow_locally()

### Community 38 - "Swift build graph: debug target client"
Cohesion: 0.33
Nodes (5): client, file-system, name, default, tools

### Community 39 - "Swift build graph: debug exe build command"
Cohesion: 0.33
Nodes (6): args, description, inputs, outputs, tool, C.CarotidSegApp-arm64-apple-macosx15.0-debug.exe

### Community 40 - "Swift build graph: debug exe entitlements"
Cohesion: 0.33
Nodes (6): args, description, inputs, outputs, tool, C.CarotidSegApp-arm64-apple-macosx15.0-debug.exe-entitlements

### Community 41 - "Swift build graph: debug module command"
Cohesion: 0.33
Nodes (6): args, description, inputs, outputs, tool, C.CarotidSegApp-arm64-apple-macosx15.0-debug.module

### Community 42 - "Swift build graph: debug PackageStructure"
Cohesion: 0.33
Nodes (6): PackageStructure, allow-missing-inputs, description, inputs, outputs, tool

### Community 43 - "Swift build graph: debug swift-version node"
Cohesion: 0.33
Nodes (6): /Users/egor/Desktop/carotid-segmentation/CarotidSegApp/.build/arm64-apple-macosx/debug/swift-version--1AB21518FC5DEDBE.txt, always-out-of-date, description, inputs, outputs, tool

### Community 44 - "Swift build graph: debug file nodes"
Cohesion: 0.33
Nodes (6): nodes, /Users/egor/Desktop/carotid-segmentation/CarotidSegApp/.build/arm64-apple-macosx/debug/CarotidSegApp, /Users/egor/Desktop/carotid-segmentation/CarotidSegApp/Sources/CarotidSegApp/, is-mutated, content-exclusion-patterns, is-directory-structure

### Community 45 - "Swift build graph: debug targets"
Cohesion: 0.33
Nodes (6): targets, CarotidSegApp-arm64-apple-macosx15.0-debug.exe, CarotidSegApp-arm64-apple-macosx15.0-debug.module, main, PackageStructure, test

### Community 46 - "Swift build graph: plugin-tools target client"
Cohesion: 0.33
Nodes (5): client, file-system, name, default, tools

### Community 47 - "Swift build graph: plugin-tools exe command"
Cohesion: 0.33
Nodes (6): args, description, inputs, outputs, tool, C.CarotidSegApp-arm64-apple-macosx15.0-debug.exe

### Community 48 - "Swift build graph: plugin-tools entitlements"
Cohesion: 0.33
Nodes (6): args, description, inputs, outputs, tool, C.CarotidSegApp-arm64-apple-macosx15.0-debug.exe-entitlements

### Community 49 - "Swift build graph: plugin-tools module command"
Cohesion: 0.33
Nodes (6): args, description, inputs, outputs, tool, C.CarotidSegApp-arm64-apple-macosx15.0-debug.module

### Community 50 - "Swift build graph: plugin-tools PackageStructure"
Cohesion: 0.33
Nodes (6): PackageStructure, allow-missing-inputs, description, inputs, outputs, tool

### Community 51 - "Swift build graph: plugin-tools swift-version node"
Cohesion: 0.33
Nodes (6): /Users/egor/Desktop/carotid-segmentation/CarotidSegApp/.build/arm64-apple-macosx/debug/swift-version--1AB21518FC5DEDBE.txt, always-out-of-date, description, inputs, outputs, tool

### Community 52 - "Swift build graph: plugin-tools file nodes"
Cohesion: 0.33
Nodes (6): nodes, /Users/egor/Desktop/carotid-segmentation/CarotidSegApp/.build/arm64-apple-macosx/debug/CarotidSegApp, /Users/egor/Desktop/carotid-segmentation/CarotidSegApp/Sources/CarotidSegApp/, is-mutated, content-exclusion-patterns, is-directory-structure

### Community 53 - "Swift build graph: plugin-tools targets"
Cohesion: 0.33
Nodes (6): targets, CarotidSegApp-arm64-apple-macosx15.0-debug.exe, CarotidSegApp-arm64-apple-macosx15.0-debug.module, main, PackageStructure, test

### Community 54 - "README anatomy/TotalSegmentator task terms"
Cohesion: 0.33
Nodes (5): Common Carotid Artery (CCA), TotalSegmentator 'headneck_bones_vessels' task, Internal Carotid Artery (ICA), TotalSegmentator 'total' task, TotalSegmentator (==2.16.0)

### Community 55 - "Swift build graph: debug build-sources command"
Cohesion: 0.40
Nodes (5): /Users/egor/Desktop/carotid-segmentation/CarotidSegApp/.build/arm64-apple-macosx/debug/CarotidSegApp.build/sources, description, inputs, outputs, tool

### Community 56 - "Swift build graph: debug entitlement-plist command"
Cohesion: 0.40
Nodes (5): /Users/egor/Desktop/carotid-segmentation/CarotidSegApp/.build/arm64-apple-macosx/debug/CarotidSegApp-entitlement.plist, description, inputs, outputs, tool

### Community 57 - "Swift build graph: debug LinkFileList command"
Cohesion: 0.40
Nodes (5): /Users/egor/Desktop/carotid-segmentation/CarotidSegApp/.build/arm64-apple-macosx/debug/CarotidSegApp.product/Objects.LinkFileList, description, inputs, outputs, tool

### Community 58 - "Swift build graph: plugin-tools build-sources command"
Cohesion: 0.40
Nodes (5): /Users/egor/Desktop/carotid-segmentation/CarotidSegApp/.build/arm64-apple-macosx/debug/CarotidSegApp.build/sources, description, inputs, outputs, tool

### Community 59 - "Swift build graph: plugin-tools entitlement-plist command"
Cohesion: 0.40
Nodes (5): /Users/egor/Desktop/carotid-segmentation/CarotidSegApp/.build/arm64-apple-macosx/debug/CarotidSegApp-entitlement.plist, description, inputs, outputs, tool

### Community 60 - "Swift build graph: plugin-tools LinkFileList command"
Cohesion: 0.40
Nodes (5): /Users/egor/Desktop/carotid-segmentation/CarotidSegApp/.build/arm64-apple-macosx/debug/CarotidSegApp.product/Objects.LinkFileList, description, inputs, outputs, tool

### Community 61 - "server app SU2 mesh job orchestration"
Cohesion: 0.50
Nodes (4): start_su2_mesh(), on_stage(), run_job(), _report()

## Knowledge Gaps
- **65 isolated node(s):** `PackageDescription`, `.id`, `seriesUID`, `description`, `modality` (+60 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 367 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **7 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `AppState` connect `AppState centerline/slice navigation` to `Swift APIClient (backend HTTP calls)`, `Boundary Conditions UI + SU2 params state`, `AppState project folder save actions`, `AppState + BoundaryConditionsView core`, `Centerline points panel UI`, `Plane pane view (slice rendering UI)`, `SurfacePreviewView (STL parsing/rendering)`, `AppState project load/save actions`, `AppState SU2 mesh/run triggers + SliceView workflow`, `Scene3DController (SceneKit lifecycle)`?**
  _High betweenness centrality (0.090) - this node is a cross-community bridge._
- **Are the 14 inferred relationships involving `AppState` (e.g. with `.body` and `.column()`) actually correct?**
  _`AppState` has 14 INFERRED edges - model-reasoned connections that need verification._
- **What connects `PackageDescription`, `.id`, `seriesUID` to the rest of the system?**
  _65 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Swift APIClient (backend HTTP calls)` be split into smaller, more focused modules?**
  _Cohesion score 0.09246575342465753 - nodes in this community are weakly interconnected._
- **Why does `APIClient` connect `Swift APIClient (backend HTTP calls)` to `AppState project load/save actions`, `AppState centerline/slice navigation`?**
  _High betweenness centrality (0.040) - this node is a cross-community bridge._
- **Should `Boundary Conditions UI + SU2 params state` be split into smaller, more focused modules?**
  _Cohesion score 0.05649717514124294 - nodes in this community are weakly interconnected._
- **Why does `AppKit` connect `AppState + BoundaryConditionsView core` to `Swift APIClient (backend HTTP calls)`, `Interactive3DSceneView orbit camera`?**
  _High betweenness centrality (0.021) - this node is a cross-community bridge._