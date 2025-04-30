import argparse
import numpy as np
import os
import copy
import pickle
import datetime
import shutil
from scipy.spatial.transform import Rotation as R
from pydrake.geometry import (
    StartMeshcat,
    RenderLabel,
    Role,
)
from pydrake.all import ConstantVectorSource, VectorLogSink
from pydrake.systems.analysis import Simulator
from pydrake.systems.framework import DiagramBuilder
from pydrake.systems.sensors import CameraInfo
from pydrake.systems.primitives import (
    PortSwitch,
    Multiplexer,
    Demultiplexer
)
from pydrake.perception import (
    DepthImageToPointCloud
)

# from manipulation.systems import AddIiwaDifferentialIK
from manipulation.systems import ExtractPose
from manipulation.station import MakeHardwareStation, LoadScenario

import pydrake.planning as mut
from pydrake.common import RandomGenerator, Parallelism, use_native_cpp_logging
from pydrake.planning import (RobotDiagramBuilder,
                              SceneGraphCollisionChecker,
                              CollisionCheckerParams)
from pydrake.math import (
    RigidTransform,
    RotationMatrix,
    RollPitchYaw,
)
from pathlib import Path
from planning.two_grasp_display_planner import PlannerState, PickState
from pydrake.solvers import MosekSolver, GurobiSolver
from pydrake.all import LeafSystem, Value, Context, InputPort
from planning.two_grasp_display_planner import TwoGraspPlanner
from planning.turntable_planner import TurntablePlanner
from perception.image_saver import ImageSaver
from perception.camera_in_world import CameraPoseInWorldSource
from planning.trajectory_sources import TrajectoryWithTimingInformationSource, DummyTrajSource
from planning.diffik import AddIiwaDifferentialIK
# from iiwa import IiwaHardwareStationDiagram

def get_regions_static(scenario_path, dirstr):
    print("generating static regions")
    use_native_cpp_logging()
    params = dict(edge_step_size=0.125)
    builder = RobotDiagramBuilder()
    builder.parser().AddModels(scenario_path)
    iiwa_model_instance_index = builder.plant().GetModelInstanceByName("iiwa")
    wsg_model_instance_index = builder.plant().GetModelInstanceByName("wsg")
    params["robot_model_instances"] = [iiwa_model_instance_index, wsg_model_instance_index]
    params["model"] = builder.Build()
    checker = SceneGraphCollisionChecker(**params)

    options = mut.IrisFromCliqueCoverOptions()
    options.num_points_per_coverage_check = 5000
    options.num_points_per_visibility_round = 1000
    options.minimum_clique_size = 16
    options.coverage_termination_threshold = 0.7

    generator = RandomGenerator(0)

    if (MosekSolver().available() and MosekSolver().enabled()) or (
            GurobiSolver().available() and GurobiSolver().enabled()):
        # We need a MIP solver to be available to run this method.
        sets = mut.IrisInConfigurationSpaceFromCliqueCover(
            checker=checker, options=options, generator=generator,
            sets=[]
        )

        if len(sets) < 1:
            raise("No regions found")
        
        time_str = datetime.datetime.now().strftime('%d%m%y_%H%M%S')
        with open(dirstr+f'/{scenario_path.split("/")[-1]}_{time_str}_regions.pkl', 'wb') as f:
            pickle.dump(sets, f)

        return sets
    else:
        print("No solvers available")

class SystemIDDataSaver(LeafSystem):
    def __init__(
            self, output_dir: str
        ):
        super().__init__()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)

        # Each array has shape (7,)
        self.measured_positions: list[np.ndarray] = []
        self.measured_torques: list[np.ndarray] = []
        self.measured_times: list[float] = []
        self.measured_wsg_positions: list[float] = []
        self.current_start_time = None
        self.object_saved = False

        self._planner_state_input_port = self.DeclareAbstractInputPort(
            "planner_state", model_value=Value(PlannerState.START))
        self._pick_state_input_port = self.DeclareAbstractInputPort(
            "pick_state", model_value=Value(PickState.IDLE))

        self._iiwa_position_input_port = self.DeclareVectorInputPort(
            "iiwa.position_measured", size=7
        )
        self._iiwa_torque_input_port = self.DeclareVectorInputPort(
            "iiwa.torque_measured", size=7
        )
        self._wsg_position_input_port = self.DeclareVectorInputPort(
            "wsg.position_measured", size=1
        )

        self.DeclarePeriodicPublishEvent(1e-3, 0.0, self.save_logs)

        
    def save_logs(self, context: Context):
        mode = self._planner_state_input_port.Eval(context)
        pick_mode = self._pick_state_input_port.Eval(context)
        if mode == PlannerState.RESET and not self.object_saved:
            self.save_to_disk()
            self.measured_positions = []
            self.measured_torques = []
            self.measured_times = []
            self.object_saved = True
            self.current_start_time = None
        elif mode == PlannerState.START:
            self.object_saved = False
        elif mode == PlannerState.SYS_ID_GRASP and pick_mode == PickState.DISPLAY:
            # Store current positions and torques.
            positions = self._iiwa_position_input_port.Eval(context)
            torques = self._iiwa_torque_input_port.Eval(context)
            wsg_position = self._wsg_position_input_port.Eval(context)

            time = context.get_time()
            if self.current_start_time is None:
                self.current_start_time = time
            traj_time = time - self.current_start_time # Want trajectory data to start at zero time
            
            self.measured_positions.append(positions)
            self.measured_torques.append(torques)
            self.measured_times.append(traj_time)
            self.measured_wsg_positions.append(wsg_position)

    def save_to_disk(self):
        if self.current_start_time is None:
            print("No system ID data to save...")
            return
        print("Saving system ID data to disk.")

        # Convert to numpy arrays.
        measured_position_data = np.stack(self.measured_positions) # Shape (T, 7)
        measured_torque_data = np.stack(self.measured_torques) # Shape (T, 7)
        sample_times_s = np.array(self.measured_times) # Shape (T,)
        measured_wsg_positions = np.array(self.measured_wsg_positions).squeeze(1) # Shape (T,)

        # Remove duplicated samples.
        _, unique_indices = np.unique(sample_times_s, return_index=True)
        if len(unique_indices) < len(sample_times_s):
            print(f"{len(unique_indices)} out of {len(sample_times_s)} data points are unique!")
            measured_position_data = measured_position_data[unique_indices]
            measured_torque_data = measured_torque_data[unique_indices]
            sample_times_s = sample_times_s[unique_indices]
            measured_wsg_positions = measured_wsg_positions[unique_indices]

        # Save to disk.
        np.save(self.output_dir / "joint_positions.npy", measured_position_data)
        np.save(self.output_dir / "joint_torques.npy", measured_torque_data)
        np.save(self.output_dir / "sample_times_s.npy", sample_times_s)
        np.save(self.output_dir / "wsg_positions.npy", measured_wsg_positions)

        print("Saved system id data to", self.output_dir)


def start_scenario(
        dirstr = "temp", 
        scenario_path="scenario_data_grasping.yml", 
        models_path="scenario_data_grasping_no_object.dmd.yaml", 
        gripper_model_path="",
        use_hardware=False, 
        save_imgs=False,
        turntable=False,
        time_horizon=10.0,
        num_objects=1
    ):

    meshcat.ResetRenderMode()

    builder = DiagramBuilder()

    dir_path = os.path.dirname(os.path.realpath(__file__))
    filename = os.path.join(dir_path, "scalable_real2sim", "pickplace_data_collection", "scenario_datas", scenario_path)
    scenario = LoadScenario(filename=filename)
    # station: IiwaHardwareStationDiagram = builder.AddNamedSystem(
    #     "station",
    #     IiwaHardwareStationDiagram(
    #         scenario=scenario, has_wsg=True, use_hardware=use_hardware
    #     ),
    # )
    models_package = os.path.abspath(os.path.join(dir_path, "scalable_real2sim", "pickplace_data_collection", "models", "package.xml"))
    station = builder.AddSystem(MakeHardwareStation(scenario, meshcat, hardware=False, package_xmls=[models_package]))
    if use_hardware:
        scenario.plant_config.time_step = 5e-3 # Controller frequency
        external_station = builder.AddSystem(MakeHardwareStation(scenario, meshcat, hardware=True, package_xmls=[models_package]))
    plant = station.GetSubsystemByName("plant")
    # plant = station.get_plant()

    # initialize image writer and save directories
    if os.path.exists(dirstr):
        shutil.rmtree(dirstr)
    os.makedirs(dirstr)
    if save_imgs:
        # save images
        if use_hardware:
            # this doesn't work because saving images takes too long and bottlenecks the whole system
            # please use scripts/realsense.py instead in parallel
            img_saver = builder.AddSystem(
                ImageSaver(
                    depth_format="32F", 
                    dirstr=dirstr, 
                    labels=False, 
                    camera_info=True,
                    use_hardware=True,
                    num_objects=num_objects
                    )
                )
        else:
            img_saver = builder.AddSystem(
                ImageSaver(
                    depth_format="32F",
                    dirstr=dirstr, 
                    labels=True, 
                    camera_info=False, 
                    ob_in_cam=True,
                    object_index=plant.GetBodyByName("base_link_mustard").index(),
                    camera_index=plant.GetBodyByName("base").index()
                )
            )
            builder.Connect(station.GetOutputPort("camera0.label_image"), img_saver.GetInputPort("label_in"))
            builder.Connect(station.GetOutputPort("camera0.rgb_image"), img_saver.GetInputPort("rgb_in"))
            builder.Connect(station.GetOutputPort("camera0.depth_image"), img_saver.GetInputPort("depth_in"))
            builder.Connect(station.GetOutputPort("body_poses"), img_saver.GetInputPort("body_poses"))

    sys_id_saver = builder.AddSystem(SystemIDDataSaver(output_dir=os.path.join(dirstr, "system_id_data")))

    # initialize point cloud output ports and save camera instrinsics
    if not use_hardware:
        camera0 = station.GetSubsystemByName("rgbd_sensor_camera0")
        camera1 = station.GetSubsystemByName("rgbd_sensor_camera1")
        camera2 = station.GetSubsystemByName("rgbd_sensor_camera2")
        camera_bin = station.GetSubsystemByName("rgbd_sensor_camera_bin")
        K = camera0.default_color_render_camera().core().intrinsics().intrinsic_matrix()
        if save_imgs:
            np.savetxt(dirstr+"/cam_K.txt", K)

        camera0_pcd = builder.AddSystem(DepthImageToPointCloud(camera0.default_depth_render_camera().core().intrinsics()))
        camera1_pcd = builder.AddSystem(DepthImageToPointCloud(camera1.default_depth_render_camera().core().intrinsics()))
        camera2_pcd = builder.AddSystem(DepthImageToPointCloud(camera2.default_depth_render_camera().core().intrinsics()))
        camera_bin_pcd = builder.AddSystem(DepthImageToPointCloud(camera_bin.default_depth_render_camera().core().intrinsics()))
        builder.Connect(station.GetOutputPort("camera0.depth_image"), camera0_pcd.GetInputPort("depth_image"))
        builder.Connect(station.GetOutputPort("camera1.depth_image"), camera1_pcd.GetInputPort("depth_image"))
        builder.Connect(station.GetOutputPort("camera2.depth_image"), camera2_pcd.GetInputPort("depth_image"))
        builder.Connect(station.GetOutputPort("camera_bin.depth_image"), camera_bin_pcd.GetInputPort("depth_image"))

    else:
        camera0_pcd = builder.AddSystem(DepthImageToPointCloud(CameraInfo(848, 480, 600.165, 600.165, 429.152, 232.822)))
        camera1_pcd = builder.AddSystem(DepthImageToPointCloud(CameraInfo(848, 480, 626.633, 626.633, 432.041, 245.465)))
        camera2_pcd = builder.AddSystem(DepthImageToPointCloud(CameraInfo(848, 480, 596.492, 596.492, 416.694, 240.225)))
        camera_bin_pcd = builder.AddSystem(DepthImageToPointCloud(CameraInfo(640, 480, 385.218, 385.218, 321.295, 244.071)))

        builder.Connect(external_station.GetOutputPort("camera0.depth_image"), camera0_pcd.GetInputPort("depth_image"))
        builder.Connect(external_station.GetOutputPort("camera1.depth_image"), camera1_pcd.GetInputPort("depth_image"))
        builder.Connect(external_station.GetOutputPort("camera2.depth_image"), camera2_pcd.GetInputPort("depth_image"))
        builder.Connect(external_station.GetOutputPort("camera_bin.depth_image"), camera_bin_pcd.GetInputPort("depth_image"))

    
    if use_hardware:
        # from camera calibation
        # Front camera
        x_front_rgb = RigidTransform(np.loadtxt("/home/real2sim/calibrations/2_10_calibrations_aligned/front.txt"))

        # Back Right camera
        x_back_right_rgb = RigidTransform(np.loadtxt("/home/real2sim/calibrations/2_10_calibrations_aligned/back_right.txt"))

        # Back Left camera
        x_back_left_rgb = RigidTransform(np.loadtxt("/home/real2sim/calibrations/2_10_calibrations_aligned/back_left.txt"))
        
        # Bin camera
        x_bin_rgb = RigidTransform(np.loadtxt("/home/real2sim/calibrations/bin_calibration_2_7_daniilidis.txt"))

        # rgb calibration to depth calibration (from realsense specs)
        # Front camera
        x_depth_rgb_front = RigidTransform([[0.999986,      -0.000127587,   0.00531376, 0.015102],
                                            [0.000116105,   0.999998,       0.00216102, 6.44158e-05],
                                            [-0.00531402,   -0.00216038,    0.999984,   -0.000426644],
                                            [0,             0,              0,          1]])
        x_front_camera = x_front_rgb @ x_depth_rgb_front

        # Back Right camera
        x_depth_rgb_back_right = RigidTransform([[0.999968,  -0.00700185,   0.00399879,     0.015085],
                                                [ 0.00701494,      0.99997,  -0.00326805,  -2.1265e-05],
                                                [-0.00397579,   0.00329599,     0.999987, -0.000455872],
                                                [          0,            0,            0,            1]])
        x_back_right_camera = x_back_right_rgb @ x_depth_rgb_back_right

        # Back Left camera
        x_depth_rgb_back_left = RigidTransform([[0.999998, -0.000191981,  -0.00215977,    0.0150991],
                                                [0.000214442,     0.999946,    0.0104041,  7.71731e-05],
                                                [0.00215765,   -0.0104046,     0.999944, -0.000317806],
                                                [          0,            0,            0,            1]])
        x_back_left_camera = x_back_left_rgb @ x_depth_rgb_back_left

        # Bin camera
        x_depth_rgb_bin = RigidTransform([[0.999968,    0.00149319,  0.00783427,    0.0147784],
                                          [-0.00146555,   0.999993, -0.00353279, -4.93721e-05],
                                          [-0.00783949, 0.00352119,    0.999963,  0.000204544],
                                          [          0,          0,           0,            1]])
        x_bin_camera = x_bin_rgb @ x_depth_rgb_bin
    else:
        # Front camera
        x_front_camera = RigidTransform(
            RotationMatrix(RollPitchYaw(-150.29508676 / 180. * np.pi, -0.49652966 / 180. * np.pi, 87.69325379 / 180. * np.pi)),
            [1.00847, -0.0314675, 1.12864]
        )

        # Back Right camera
        x_back_right_camera = RigidTransform(
            RotationMatrix(RollPitchYaw(-105.81290946 / 180. * np.pi, 2.14985993, -43.7254432 / 180. * np.pi)),
            [-0.110748, -0.931772,  0.388191]
        )

        # Back Left camera
        x_back_left_camera = RigidTransform(
            RotationMatrix(RollPitchYaw(-102.739428 / 180. * np.pi, -3.69469624 / 180. * np.pi, -149.1420755 / 180. * np.pi)),
            [-0.0533544,  1.00955,  0.449207]
        )

        # Bin camera
        x_bin_camera = RigidTransform(
            RotationMatrix(RollPitchYaw(-164.69831287 / 180. * np.pi, -35.83297034 / 180. * np.pi, -99.44115857 / 180. * np.pi)),
            [-0.0574518,  0.874365 ,  0.332985]
        )

    # connect stationary camera pcd source
    camera0_pose_source = builder.AddSystem(CameraPoseInWorldSource(x_front_camera, handeye=False))
    camera1_pose_source = builder.AddSystem(CameraPoseInWorldSource(x_back_right_camera, handeye=False))
    camera2_pose_source = builder.AddSystem(CameraPoseInWorldSource(x_back_left_camera, handeye=False))
    bin_cam_pose_source = builder.AddSystem(CameraPoseInWorldSource(x_bin_camera, handeye=False))

    builder.Connect(
        camera0_pose_source.GetOutputPort("X_WC"),
        camera0_pcd.GetInputPort("camera_pose"),
    )

    builder.Connect(
        camera1_pose_source.GetOutputPort("X_WC"),
        camera1_pcd.GetInputPort("camera_pose"),
    )

    builder.Connect(
        camera2_pose_source.GetOutputPort("X_WC"),
        camera2_pcd.GetInputPort("camera_pose"),
    )

    builder.Connect(
        bin_cam_pose_source.GetOutputPort("X_WC"),
        camera_bin_pcd.GetInputPort("camera_pose"),
    )

    controller_plant = station.GetSubsystemByName(
        "iiwa_controller_plant_pointer_system"
    ).get()

    # Set up planner
    if turntable:
        planner = builder.AddSystem(TurntablePlanner(
                plant=plant, 
                controller_plant=controller_plant,
                X_WC0=x_front_camera,
                X_WC1=x_back_left_camera,
                X_WC2=x_back_right_camera,
                X_WC_bin=x_bin_camera,
                meshcat=meshcat,
                dirstr=dirstr,
                models_path=os.path.join(dir_path, "scalable_real2sim", "pickplace_data_collection", "scenario_datas", models_path),
                gripper_model_path=gripper_model_path))
    else:
        planner = builder.AddSystem(
            TwoGraspPlanner(
                plant=plant, 
                controller_plant=controller_plant,
                X_WC0=x_front_camera,
                X_WC1=x_back_left_camera,
                X_WC2=x_back_right_camera,
                X_WC_bin=x_bin_camera,
                meshcat=meshcat,
                dirstr=dirstr,
                time_horizon=time_horizon,
                models_path=os.path.join(dir_path, "scalable_real2sim", "pickplace_data_collection", "scenario_datas", models_path),
                gripper_model_path=gripper_model_path,
                num_objs=num_objects
            )
        )

    if save_imgs:
        builder.Connect(planner.GetOutputPort("planner_state"), img_saver.GetInputPort("planner_state"))

    wsg_state_demux: Demultiplexer = builder.AddSystem(Demultiplexer(2, 1))
    if use_hardware:
        # Connect the output of external station to the input of internal station
        builder.Connect(
            external_station.GetOutputPort("iiwa.position_measured"),
            station.GetInputPort("iiwa.position"),
        )

        builder.Connect(
            external_station.GetOutputPort("wsg.state_measured"),
            wsg_state_demux.get_input_port(),
        )
        builder.Connect(
            wsg_state_demux.get_output_port(0),
            station.GetInputPort("wsg.position"),
        )
        builder.Connect(
            wsg_state_demux.get_output_port(0),
            planner.GetInputPort("wsg.position_measured"),
        )
    else:
        builder.Connect(
            station.GetOutputPort("wsg.state_measured"),
            wsg_state_demux.get_input_port(),
        )
        builder.Connect(
            wsg_state_demux.get_output_port(0),
            planner.GetInputPort("wsg.position_measured"),
        )


    # Connect system ID data saver ports.
    builder.Connect(planner.GetOutputPort("planner_state"), sys_id_saver.GetInputPort("planner_state"))
    builder.Connect(planner.GetOutputPort("pick_state"), sys_id_saver.GetInputPort("pick_state"))
    if not use_hardware:
        builder.Connect(
            station.GetOutputPort("iiwa.position_measured"),
            sys_id_saver.GetInputPort("iiwa.position_measured"),
        )
        builder.Connect(
            station.GetOutputPort("iiwa.torque_measured"),
            sys_id_saver.GetInputPort("iiwa.torque_measured"),
        )
    else:
        builder.Connect(
            external_station.GetOutputPort("iiwa.position_measured"),
            sys_id_saver.GetInputPort("iiwa.position_measured"),
        )
        builder.Connect(
            external_station.GetOutputPort("iiwa.torque_measured"),
            sys_id_saver.GetInputPort("iiwa.torque_measured"),
        )

    builder.Connect(
        wsg_state_demux.get_output_port(0),
        sys_id_saver.GetInputPort("wsg.position_measured"),
    )
    
    if not use_hardware:
        builder.Connect(
            station.GetOutputPort("iiwa.position_measured"),
            planner.GetInputPort("iiwa_position"),
        )
    else:
        builder.Connect(
            external_station.GetOutputPort("iiwa.position_measured"),
            planner.GetInputPort("iiwa_position"),
        )

    joint_traj_source: TrajectoryWithTimingInformationSource = (
        builder.AddNamedSystem(
            "joint_traj_source",
            TrajectoryWithTimingInformationSource(
                trajectory_size=7
            ),
        )
    )

    builder.Connect(
        planner.GetOutputPort("joint_position_trajectory"),
        joint_traj_source.GetInputPort("trajectory"),
    )
    if use_hardware:
        builder.Connect(
            external_station.GetOutputPort("iiwa.position_commanded"),
            joint_traj_source.GetInputPort("current_cmd"),
        )
    else:
        builder.Connect(
            station.GetOutputPort("iiwa.position_measured"),
            joint_traj_source.GetInputPort("current_cmd"),
        )

    if use_hardware:
        builder.Connect(
            planner.GetOutputPort("wsg_position"),
            external_station.GetInputPort("wsg.position"),
        )
    else:
        builder.Connect(
            planner.GetOutputPort("wsg_position"),
            station.GetInputPort("wsg.position"),
        )

    # Increase max force.
    wsg_force_source = builder.AddNamedSystem(
        "wsg_force_source", ConstantVectorSource([80.0]) # 80N is max
    )
    builder.Connect(
        wsg_force_source.get_output_port(), station.GetInputPort("wsg.force_limit")
    )

    # Set up logging for system ID.
    num_positions = 7
    measured_position_logger: VectorLogSink = builder.AddNamedSystem(
        "measured_position_logger",
        VectorLogSink(num_positions, publish_period=scenario.plant_config.time_step),
    )
    builder.Connect(
        station.GetOutputPort("iiwa.position_measured"),
        measured_position_logger.get_input_port(),
    )
    measured_torque_logger: VectorLogSink = builder.AddNamedSystem(
        "measured_torque_logger",
        VectorLogSink(num_positions, publish_period=scenario.plant_config.time_step),
    )
    builder.Connect(
        station.GetOutputPort("iiwa.torque_measured"),
        measured_torque_logger.get_input_port(),
    )

    # Set up differential inverse kinematics.
    velocity_limits = 0.2 * np.ones(7)
    acceleration_limits = 0.1 * np.ones(7)
    diff_ik = AddIiwaDifferentialIK(
        builder, 
        controller_plant, 
        frame=None,
        velocity_lims=velocity_limits, 
        acceleration_lims=acceleration_limits, # doesn't actually do anything since using this stops the robot from moving???
        joint_centering_gain=1.0
    )
    builder.Connect(planner.GetOutputPort("X_WG"), diff_ik.get_input_port(0))
    if use_hardware:
        builder.Connect(
            external_station.GetOutputPort("iiwa.state_estimated"),
            diff_ik.GetInputPort("robot_state"),
        )
    else:
        builder.Connect(
            station.GetOutputPort("iiwa.state_estimated"),
            diff_ik.GetInputPort("robot_state"),
        )
    builder.Connect(
        planner.GetOutputPort("reset_diff_ik"),
        diff_ik.GetInputPort("use_robot_state"),
    )

    # The DiffIK and the direct position-control modes go through a PortSwitch
    switch = builder.AddSystem(PortSwitch(7))
    builder.Connect(diff_ik.get_output_port(), switch.DeclareInputPort("diff_ik"))
    builder.Connect(
        joint_traj_source.get_output_port(),
        switch.DeclareInputPort("position"),
    )
    if use_hardware:
        builder.Connect(switch.get_output_port(), external_station.GetInputPort("iiwa.position"))
    else:
        builder.Connect(switch.get_output_port(), station.GetInputPort("iiwa.position"))
    builder.Connect(
        planner.GetOutputPort("control_mode"),
        switch.get_port_selector_input_port(),
    )

    builder.Connect(
        camera0_pcd.GetOutputPort("point_cloud"),
        planner.GetInputPort("cloud_front"),
    )
    builder.Connect(
        camera1_pcd.GetOutputPort("point_cloud"),
        planner.GetInputPort("cloud_back_left"),
    )
    builder.Connect(
        camera2_pcd.GetOutputPort("point_cloud"),
        planner.GetInputPort("cloud_back_right"),
    )
    builder.Connect(
        camera_bin_pcd.GetOutputPort("point_cloud"),
        planner.GetInputPort("cloud_bin"),
    )
    builder.Connect(
        station.GetOutputPort("body_poses"),
        planner.GetInputPort("body_poses"),
    )

    # Build diagram
    diagram = builder.Build()
    context = diagram.CreateDefaultContext()

    # Simulate
    simulator = Simulator(diagram)
    simulator_context = simulator.get_mutable_context()

    # Remove labels of anything but mustard and gripper
    if not use_hardware:
        scene_graph = station.GetSubsystemByName("scene_graph")
        source_id = plant.get_source_id()
        scene_graph_context = scene_graph.GetMyMutableContextFromRoot(simulator_context)
        query_object = scene_graph.get_query_output_port().Eval(scene_graph_context)
        inspector = query_object.inspector()
        for geometry_id in inspector.GetAllGeometryIds():
            properties = copy.deepcopy(inspector.GetPerceptionProperties(geometry_id))
            if properties is None:
                continue
            frame_id = inspector.GetFrameId(geometry_id)
            body = plant.GetBodyFromFrameId(frame_id)
            if body.model_instance() == plant.GetModelInstanceByName("mustard_bottle"):
                properties.UpdateProperty("label", "id", RenderLabel(0)) # Make mustard label 0
            elif body.model_instance() == plant.GetModelInstanceByName("wsg"):
                properties.UpdateProperty("label", "id", RenderLabel(1)) # Make gripper label 1
            else:
                properties.UpdateProperty("label", "id", RenderLabel.kDontCare)
            scene_graph.RemoveRole(scene_graph_context, source_id, geometry_id, Role.kPerception)
            scene_graph.AssignRole(scene_graph_context, source_id, geometry_id, properties)

    simulator.set_target_realtime_rate(1.0)

    meshcat.AddButton("Stop Simulation", "Escape")
    print("Press Escape to stop the simulation")
    while meshcat.GetButtonClicks("Stop Simulation") < 1 and not planner.done:
        simulator.AdvanceTo(simulator.get_context().get_time() + 5000.0)

    meshcat.DeleteButton("Stop Simulation")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scenario_path",
        default="scenario_data_grasping.yml",
        help="yaml file with scenario",
    )
    parser.add_argument(
        "--models_path",
        default="scenario_data_grasping.dmd.yaml",
        help="dmd.yaml file with scenario, used for checking for collisions",
        nargs='?',
    )
    parser.add_argument(
        "--save_dir",
        default="temp",
        help="directory to save images in",
        nargs='?',
    )
    parser.add_argument(
        "--num_objects",
        default="1",
        help="number of objects to scan",
        nargs='?',
    )
    parser.add_argument(
        "--use_hardware",
        action="store_true",
        help="Whether to use real world hardware.",
    )
    parser.add_argument(
        "--save_imgs",
        action='store_true',
        help="yaml file with scenario",
    )
    parser.add_argument(
        "--use_custom_path_planner",
        action="store_true",
        help="Whether to use user implemented path planner.",
    )
    parser.add_argument(
        "--turntable",
        action="store_true",
        help="Whether to use turntable planner.",
    )
    parser.add_argument(
        "--time_horizon",
        type=float,
        default=10.0,
        help="The time horizon/ duration of the trajectory. Only used for Fourier "
        + "series trajectories.",
    )
    args = parser.parse_args()

    directory_path = os.path.dirname(os.path.abspath(__file__))
    gripper_model_path = "package://pickplace_data_collection/schunk_wsg_50_large_grippers_w_buffer.sdf"

    # Start the visualizer.
    meshcat = StartMeshcat()

    save_dir_path = os.path.abspath(os.path.join(os.path.dirname( __file__ ), 'tests', args.save_dir))
    start_scenario(
        save_dir_path, 
        scenario_path= args.scenario_path, 
        gripper_model_path=gripper_model_path,
        models_path=args.models_path, 
        use_hardware=args.use_hardware, 
        save_imgs=args.save_imgs,
        num_objects=int(args.num_objects),
        turntable=args.turntable,
        time_horizon=args.time_horizon,
    )
