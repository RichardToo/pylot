import numpy as np
import matplotlib.pyplot as plt
import cv2
import carla
import time
import random

from config import global_config
from lanenet_model import lanenet
from lanenet_model import lanenet_postprocess


class LanePredictor():
    def __init__(self, weights, config):
        """ Initializes a LanePredictor which is used to register a callback
        for the RGB images and predict lanes.

        Args:
            weights: The path of the weights to be used in the prediction.
            config: The config to be used for tensorflow.
        """
        self.input_tensor = tf.placeholder(dtype=tf.float32,
                                           shape=[1, 256, 512, 3],
                                           name='input_tensor')
        self.net = lanenet.LaneNet(phase='test', net_flag='vgg')
        self.binary_seg_ret, self.instance_seg_ret = net.inference(
            input_tensor=input_tensor, name='lanenet_model')

        self.postprocessor = lanenet_postprocess.LaneNetPostProcessor()
        sess_config = tf.ConfigProto()
        sess_config.gpu_options.per_process_gpu_memory_fraction = \
                config.TEST.GPU_MEMORY_FRACTION
        sess_config.gpu_options.allow_growth = config.TRAIN.TF_ALLOW_GROWTH
        sess_config.gpu_options.allocator_type = 'BFC'
        self.sess = tf.Session(config=sess_config).__enter__()
        saver = tf.train.Saver()
        saver.restore(sess=self.sess, save_path=weights)

    def process_images(self, msg):
        """ This function runs the LaneNet model on each of the image retrieved
        from the simulator.

        Args:
            msg: The RGB image received from the camera.
        """
        # Convert the BGRA image to BGR.
        image = np.frombuffer(msg.raw_data, dtype=np.dtype('uint8'))
        image = np.reshape(image, (msg.height, msg.width, 4))[:, :, :3]
        image_vis = image
        image = cv2.resize(image, (512, 256), interpolation=cv2.INTER_LINEAR)
        resized_image = image
        image = image / 127.5 - 1.0

        binary_seg_image, instance_seg_image = self.sess.run(
            [binary_seg_ret, instance_seg_ret],
            feed_dict={input_tensor: [image]})

        postprocess_result = postprocessor.postprocess(
            binary_seg_result=binary_seg_image[0],
            instance_seg_result=instance_seg_image[0],
            source_image=image_vis)
        mask_image = postprocess_result['mask_image']

        cv2.imshow(
            "Lane Detection",
            cv2.addWeighted(resized_image[:, :, (2, 1, 0)], 1,
                            mask_image[:, :, (2, 1, 0)], 0.3, 0))


def spawn_driving_vehicle(client, world):
    """ This function spawns the driving vehicle and puts it into
    an autopilot mode.

    Args:
        client: The carla.Client instance representing the simulation to
          connect to.
        world: The world inside the current simulation.

    Returns:
        A carla.Actor instance representing the vehicle that was just spawned.
    """
    # Get the blueprint of the vehicle and set it to AutoPilot.
    vehicle_bp = random.choice(
        world.get_blueprint_library().filter('vehicle.*'))
    while not vehicle_bp.has_attribute('number_of_wheels') or not int(
            vehicle_bp.get_attribute('number_of_wheels')) == 4:
        vehicle_bp = random.choice(
            world.get_blueprint_library().filter('vehicle.*'))
    vehicle_bp.set_attribute('role_name', 'autopilot')

    # Get the spawn point of the vehicle.
    start_pose = random.choice(world.get_map().get_spawn_points())

    # Spawn the vehicle.
    batch = [
        carla.command.SpawnActor(vehicle_bp, start_pose).then(
            carla.command.SetAutopilot(carla.command.FutureActor, True))
    ]
    vehicle_id = client.apply_batch_sync(batch)[0].actor_id

    # Find the vehicle and return the carla.Actor instance.
    time.sleep(
        0.5)  # This is so that the vehicle gets registered in the actors.
    return world.get_actors().find(vehicle_id)


def spawn_rgb_camera(world, location, rotation, vehicle):
    """ This method spawns an RGB camera with the default parameters and the
    given location and rotation. It also attaches the camera to the given
    actor.

    Args:
        world: The world inside the current simulation.
        location: The carla.Location instance representing the location where
          the camera needs to be spawned with respect to the vehicle.
        rotation: The carla.Rotation instance representing the rotation of the
          spawned camera.
        vehicle: The carla.Actor instance to attach the camera to.

    Returns:
        An instance of the camera spawned in the world.
    """
    camera_bp = world.get_blueprint_library().find('sensor.camera.rgb')
    camera_bp.set_attribute('image_size_x', '1280')
    camera_bp.set_attribute('image_size_y', '720')
    transform = carla.Transform(location=location, rotation=rotation)
    return world.spawn_actor(camera_bp, transform, attach_to=vehicle)


def main(lane_predictor):
    # Connect to the CARLA instance.
    client = carla.Client('localhost', 2000)
    world = client.get_world()

    # Spawn the vehicle.
    vehicle = spawn_driving_vehicle(client, world)

    # Spawn the camera and register a function to listen to the images.
    camera = spawn_rgb_camera(world, carla.Location(x=2.0, y=0.0, z=1.8),
                              carla.Rotation(roll=0, pitch=0, yaw=0), vehicle)
    camera.listen(lane_predictor.process_images)

    return vehicle, camera, world


if __name__ == "__main__":
    vehicle, camera, world = main(
        LanePredictor("~/code/lanenet/model/tusimple_lanenet_vgg.cpkt",
                      global_config.cfg))
    try:
        while True:
            time.sleep(1 / 100.0)
            cv2.waitKey(1)
    except KeyboardInterrupt:
        # Destroy the actors.
        vehicle.destroy()
        camera.destroy()
