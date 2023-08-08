import rospy
import requests
from std_msgs.msg import Header
from sensor_msgs.msg import NavSatFix, NavSatStatus, Image
import pandas as pd
from datetime import datetime
import os
import numpy as np
import cv_bridge
import cv2


class Node:
    def __init__(self):
        rospy.init_node("fake_ifcb")
        rospy.loginfo("Initialized node %s.", rospy.get_name())
        self.base_url = rospy.get_param("~ifcb_dashboard_url")
        rospy.logdebug("Using base URL %s.", self.base_url)
        self.dataset = rospy.get_param("~ifcb_dataset")
        rospy.logdebug("Using dataset %s.", self.dataset)
        self.gps_fix_topic = rospy.get_param("~gps_fix_topic")
        rospy.logdebug("Using gps_fix topic %s.", self.gps_fix_topic)
        self.metadata = pd.read_csv(
            os.path.join(self.base_url, "api", "export_metadata", self.dataset),
            parse_dates=["sample_time"],
        )
        rospy.logdebug(
            "Successfully read metadata file. Found %d bins.", len(self.metadata)
        )
        self.dt_index = pd.DatetimeIndex(self.metadata.sample_time.to_list())
        self.roi_publisher = rospy.Publisher(
            rospy.get_param("~topic") + "/roi/image", Image, queue_size=10
        )
        self.gps_subscriber = rospy.Subscriber(
            self.gps_fix_topic, NavSatFix, self.gps_subscriber_callback, queue_size=20
        )
        self.cvbridge = cv_bridge.CvBridge()
        self.current_bin = None

    def nearest_bin(self, t):
        return list(self.metadata.pid[self.dt_index.get_indexer([t], method="pad")])[0]

    def roi_publisher_callback(self, header: Header):
        target_ids = pd.read_json(
            os.path.join(self.base_url, "api", "list_images", self.current_bin)
        )["images"].to_list()
        rospy.logdebug(
            "Found %d targets for bin %s.", len(target_ids), self.current_bin
        )
        for target_id in target_ids:
            rospy.logdebug(
                "Requesting target %d for bin %s...", target_id, self.current_bin
            )
            response = requests.get(
                os.path.join(
                    self.base_url,
                    self.dataset,
                    "{}_{}.jpg".format(self.current_bin, target_id),
                ),
                stream=True,
            ).raw
            arr = np.asarray(bytearray(response.read()), dtype="uint8")
            image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            msg = self.cvbridge.cv2_to_imgmsg(image, header=header)
            self.roi_publisher.publish(msg)

    def gps_subscriber_callback(self, msg):
        t = pd.to_datetime(msg.header.stamp.secs, unit="s", utc=True)
        rospy.logdebug("Received GPS fix with time %s.", str(t))
        potential_bin = self.nearest_bin(t)
        if potential_bin != self.current_bin:
            rospy.loginfo("Got new bin. Acquiring images...")
            self.current_bin = potential_bin
            self.roi_publisher_callback(msg.header)
        else:
            rospy.logdebug(
                "GPS fix does not represent new bin. No images will be acquired."
            )


def main():
    node = Node()
    rospy.spin()


if __name__ == "__main__":
    main()
