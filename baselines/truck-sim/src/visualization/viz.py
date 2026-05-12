import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

from src.simulator.common import TruckStatus, RequestStatus
from src.simulator.location import Depot, Customer
from src.simulator.request import Request
from src.simulator.truck import Truck


class Viz(object):
    """
    Visualization class that shares
    """

    def __init__(self,
                 depots: list[Depot],
                 customers: list[Customer],
                 trucks: list[Truck],
                 requests: list[Request],
                 end_time: float,
                 map_size: tuple[int, int]):

        self.depots: list[Depot] = depots
        self.customers: list[Customer] = customers
        self.trucks: list[Truck] = trucks
        self.requests: list[Request] = requests
        self.end_time: float = end_time
        self.map_size: tuple[int, int] = map_size

        colors = plt.get_cmap('hsv')
        self._truck_colors = [colors(i) for i in np.linspace(0, 1, len(self.trucks) + 1)]
        self._truck_colors.pop(-1)

        self.fig, (self.ax, self.bar) = plt.subplots(1, 2)
        self.fig.set_size_inches(5 + 10, 10)
        self.frame_idx = 0

        # self.update(0.0)

    def _draw_locations(self):
        # every depot is depicted as red squares
        for depot in self.depots:
            self.ax.plot(depot.pos.x, depot.pos.y, 'sr')

        # customers are depicted as black stars
        # - if the customer as a truck at it then it will turn red
        # for customer in self.customers:
        #     if len(customer._trucks) == 0:
        #         self.ax.plot(customer.pos.x, customer.pos.y, '*k')  # black star
        #     else:
        #         self.ax.plot(customer.pos.x, customer.pos.y, '*r')  # red star when unloading

    def _draw_trucks(self):
        for truck, color in zip(self.trucks, self._truck_colors):
            # trucks are depicted as blue dots
            if truck.status == TruckStatus.MOVING:
                self.ax.plot(truck.pos.x, truck.pos.y, '.', c=color, markersize=15)  # blue dot

            # the request the truck is actively working on is drawn via a blue line
            if truck.status != TruckStatus.IDLE:
                assert truck.request is not None
                px, py = (truck.pos.x, truck.pos.y)
                for i, to in enumerate(truck.request._route):
                    cx, cy = to.location.pos.x, to.location.pos.y

                    if i == 0 and truck.request.status == RequestStatus.STARTED:
                        self.ax.plot([px, cx], [py, cy], '--', c=color, linewidth=1)
                    else:
                        self.ax.plot([px, cx], [py, cy], c=color, linewidth=1)

                    px, py = cx, cy

    def _draw_requests(self):
        display_customers: set[Customer] = set()
        for request in self.requests:
            src_x, src_y = request.source.pos.x, request.source.pos.y
            dst_x, dst_y = request.destination.pos.x, request.destination.pos.y

            # early orders as green and late orders as red
            if request.status == RequestStatus.REJECTED:
                self.ax.plot([src_x, dst_x], [src_y, dst_y], 'tab:grey-', linewidth=1)

            elif request.status != RequestStatus.UNAVAILABLE:
                display_customers.add(request.destination)
                if request.available_time < 0:
                    self.ax.plot([src_x, dst_x], [src_y, dst_y], 'g-', linewidth=1)
                else:
                    self.ax.plot([src_x, dst_x], [src_y, dst_y], 'r-', linewidth=1)

            else:
                continue

            ar = FancyArrowPatch(
                (src_x, src_y),
                ((src_x + dst_x)/2, (src_y + dst_y)/2),
                arrowstyle='->', mutation_scale=10)
            self.ax.add_patch(ar)

        for truck, color in zip(self.trucks, self._truck_colors):
            requests: list[Request] = []
            if truck.request is not None:
                # requests.append(truck.request)
                display_customers.add(truck.request.destination)
            for request in truck.requests:
                requests.append(request)

            for request in requests:
                display_customers.add(request.destination)
                src_x, src_y = request.source.pos.x, request.source.pos.y
                dst_x, dst_y = request.destination.pos.x, request.destination.pos.y

                self.ax.plot([src_x, dst_x], [src_y, dst_y], '-', c=color, linewidth=1, alpha=0.2)
                # ar = FancyArrowPatch(
                #     (src_x, src_y),
                #     ((src_x + dst_x)/2, (src_y + dst_y)/2),
                #     arrowstyle='->', mutation_scale=10, alpha=0.2)
                # self.ax.add_patch(ar)

        for customer in display_customers:
            if len(customer._trucks) == 0:
                self.ax.plot(customer.pos.x, customer.pos.y, '*k', markersize=8)  # black star
            else:
                self.ax.plot(customer.pos.x, customer.pos.y, '*r', markersize=8)  # red star when unloading

    def _draw_schedule(self, time: float):
        for i, (truck, color) in enumerate(zip(self.trucks, self._truck_colors)):
            schedule: list[tuple[int, float, float]] = truck.get_request_schedule()
            # schedule is a list of triples where
            # - first: id for the request
            # - second: start _time of the request
            # - third: duration of the request

            # if i == 0:
            #     print(schedule)

            color_copy = np.array(color)
            color_copy[3] = 0.8
            self.bar.broken_barh(
                [(x1, x2) for _, x1, x2 in schedule],
                (5 + i*4, 3), facecolors=(color, color_copy),)
            # (5 + i * 4, 3), facecolors = ('deepskyblue', 'dodgerblue'))

            for request_id, x1, x2 in schedule:
                self.bar.text(
                    x=x1 + x2/2,
                    y=6.5 + i*4,
                    s=request_id, ha='center', va='center', color='black')

        self.bar.set_yticks(
            [6.5 + i*4 for i in range(len(self.trucks))],
            labels=[f"Truck {truck.id}" for truck in self.trucks])
        # for i, truck in enumerate(self.trucks):
        #     self.bar.set_yticks([6.5 + i*4], c=self._truck_colors[i], labels=[f"Truck {truck.id}"])

        self.bar.set_ylim(0, len(self.trucks)*4 + 9)
        self.bar.set_xlim(0, self.end_time)
        self.bar.axvline(x=time, color="black", linestyle="-")
        # self.bar.grid(True)

    def update(self, time: float):
        # TODO: optimize by using numpy and blit=True to reduce redrawing and extra calls to ax.plot
        #  https://jakevdp.github.io/blog/2012/08/18/matplotlib-animation-tutorial/
        #  https://stackoverflow.com/questions/33287156/specify-color-of-each-point-in-3d-scatter-plot
        #  https://spatialthoughts.com/2022/01/14/animated-plots-with-matplotlib/
        #  https://matplotlib.org/stable/users/explain/animations/animations.html
        #  (use a already made library instead of doing this)

        self.ax.clear()
        self.bar.clear()

        self.fig.suptitle(f"{time/60} hours", fontsize=16)
        self._draw_locations()
        self._draw_trucks()
        self._draw_requests()
        self._draw_schedule(time)

        # self.ax.figure.set_size_inches(5, 10)
        # self.bar.figure.set_size_inches(10, 10)
        self.ax.set_xlim([0, self.map_size[0]])
        self.ax.set_ylim([0, self.map_size[1]])
