import rclpy

import queue
import tf2_ros
from message_filters import TimeSynchronizer, Subscriber, SimpleFilter
import sensor_msgs


class Tf2MessageFilter(SimpleFilter):
    """Stores a message unless corresponding transforms is
    available
    """

    def __init__(self, node, fs, base_frame, target_frame,
                 queue_size=500):
        SimpleFilter.__init__(self)
        tss = TimeSynchronizer(fs, queue_size)
        self.connectInput(tss)
        self.base_frame = base_frame
        self.target_frame = target_frame
        # TODO: Use a better data structure
        self.message_queue = queue.Queue(maxsize=queue_size)
        self.tfBuffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tfBuffer, node)
        self.max_queue_size = queue_size
        self._max_queue_size_so_far = 0

    def connectInput(self, tss):
        self.incoming_connection = tss.registerCallback(
            self.input_callback)

    def poll_transforms(self, latest_msg_tstamp):
        """
        Poll transforms corresponding to all messages. If found throw older
        messages than the timestamp of transform just found
        and if not found keep all the messages.
        """
        # Check all the messages for transform availability
        tmp_queue = queue.Queue(self.max_queue_size)
        first_iter = True
        # Loop from old to new
        while not self.message_queue.empty():
            (depth_msg, rgb_msg) = self.message_queue.get()
            tstamp = depth_msg.header.stamp
            if (first_iter and
                    self.message_queue.qsize() > self._max_queue_size_so_far):
                first_iter = False
                self._max_queue_size_so_far = self.message_queue.qsize()
                print("Queue(%d) time range: %f - %f" %
                      (self.message_queue.qsize(),
                       tstamp.sec, latest_msg_tstamp.sec))
                print("Maximum queue size used:%d" %
                      self._max_queue_size_so_far)
            if self.tfBuffer.can_transform(self.base_frame, self.target_frame,
                                           tstamp):
                tf_msg = self.tfBuffer.lookup_transform(self.base_frame,
                                                        self.target_frame, tstamp)
                self.signalMessage(depth_msg, rgb_msg, tf_msg)
                # Note that we are deliberately throwing away the messages
                # older than transform we just received
                return
            else:
                # if we don't find any transform we will have to recycle all
                # the messages
                tmp_queue.put((depth_msg, rgb_msg))
        self.message_queue = tmp_queue

    def input_callback(self, depth_msg, rgb_msg):
        """ Handles incoming message """
        if self.message_queue.full():
            # throw away the oldest message
            print("Queue too small. If you this message too often"
                  + " consider increasing queue_size")
            self.message_queue.get()

        self.message_queue.put((depth_msg, rgb_msg))
        # This can be part of another timer thread
        # TODO: call this only when a new/changed transform
        self.poll_transforms(depth_msg.header.stamp)
