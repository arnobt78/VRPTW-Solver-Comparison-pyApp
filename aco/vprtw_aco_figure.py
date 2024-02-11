import matplotlib.pyplot as plt
from multiprocessing import Queue as MPQueue


class VrptwAcoFigure:
    def __init__(self, nodes: list, path_queue: MPQueue):
        """
        matplotlib drawing calculations need to be placed on the main thread. It is recommended to open another thread for path finding work.
        When the path-finding thread finds a new path, it puts the path in path_queue, and the graphics drawing thread will automatically draw it.
        The path stored in the queue exists in the form of PathMessage (class)
        The nodes stored in nodes exist in the form of Node (class). Node.x and Node.y are mainly used to obtain the coordinates of the nodes.

        :param nodes: nodes is a list of each node, including depot
        :param path_queue: The queue is used to store the path calculated by the working thread. Each element in the queue is a path, and the path stores the ID of each node.
        """

        self.nodes = nodes
        self.figure = plt.figure(figsize=(10, 10))
        self.figure_ax = self.figure.add_subplot(1, 1, 1)
        self.path_queue = path_queue
        self._depot_color = "k"
        self._customer_color = "steelblue"
        self._line_color = "darksalmon"

    def _draw_point(self):
        # draw depot
        self.figure_ax.scatter(
            [self.nodes[0].x],
            [self.nodes[0].y],
            c=self._depot_color,
            label="depot",
            s=40,
        )

        # draw customer
        self.figure_ax.scatter(
            list(node.x for node in self.nodes[1:]),
            list(node.y for node in self.nodes[1:]),
            c=self._customer_color,
            label="customer",
            s=20,
        )
        # plt.pause(0.5)

    def run(self):
        # First draw each node
        self._draw_point()
        # self.figure.show()

        # Read the new path from the queue and draw it
        # while True:
        #     if not self.path_queue.empty():
        #         # Take the latest path in the queue and discard the other paths.
        #         info = self.path_queue.get()
        #         while not self.path_queue.empty():
        #             info = self.path_queue.get()

        #         path, distance, used_vehicle_num = info.get_path_info()
        #         if path is None:
        #             print("[draw figure]: exit")
        #             break

        #         # You need to record the line to be removed first. You cannot remove it directly in the first loop.
        #         # Otherwise, self.figure_ax.lines will change during the loop, causing some lines to fail to be removed successfully.
        #         remove_obj = []
        #         for line in self.figure_ax.lines:
        #             if line._label == "line":
        #                 remove_obj.append(line)

        #         for line in remove_obj:
        #             self.figure_ax.lines.remove(line)
        #         remove_obj.clear()

        #         # redraw line
        #         self.figure_ax.set_title(
        #             "travel distance: %0.2f, number of vehicles: %d "
        #             % (distance, used_vehicle_num)
        #         )
        #         self._draw_line(path)
        #     plt.pause(1)

    def _draw_line(self, path):
        # Draw the path according to the index in the path
        for i in range(1, len(path)):
            x_list = [self.nodes[path[i - 1]].x, self.nodes[path[i]].x]
            y_list = [self.nodes[path[i - 1]].y, self.nodes[path[i]].y]
            self.figure_ax.plot(
                x_list, y_list, color=self._line_color, linewidth=1.5, label="line"
            )
            plt.pause(0.2)
