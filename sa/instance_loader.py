from math import ceil
import random

from sa.util import distance


class Customer:
    def __init__(self, cust_no, x, y, demand, ready_time, due_date, service_time):
        self.cust_no = cust_no
        self.x = x
        self.y = y
        self.demand = demand
        self.ready_time = ready_time
        self.due_date = due_date
        self.service_time = service_time
        self.is_served = False
        self.vehicle_num = None

    def copy(self):
        return Customer(self.cust_no, self.x, self.y, self.demand, self.ready_time, self.due_date, self.service_time)

    def served(self, vehicle_num):
        self.is_served = True
        self.vehicle_num = vehicle_num

    def unserve(self):
        self.is_served = False
        self.vehicle_num = None

    def __str__(self):
        return f'Customer NO. : {self.cust_no}; X : {self.x}; Y : {self.y}; Demand : {self.demand}; Ready Time : {self.ready_time}; Due Date : {self.due_date}; Service Time : {self.service_time}; Vehichle num: {self.vehicle_num}'

    def __eq__(self, other):
        return self.cust_no == other.cust_no

class Vehicle:
    def __init__(self, id, depo, max_capacity, min_capacity=0):
        self.id = id
        self.x = depo.x
        self.y = depo.y
        self.max_capacity = max_capacity  
        self.min_capacity = min_capacity
        self.capacity = max_capacity
        self.last_service_time = 0
        self.service_route = [(depo, 0)]
        self.total_distance = 0
        self.depo = depo

    def serve_customer(self, customer):
        if not (customer.ready_time <= (ceil(distance(customer, self)) + self.last_service_time) <= customer.due_date) or self.capacity <= customer.demand:
            return False
        if self.depo.due_date < ceil(distance(customer, self)) + self.last_service_time + customer.service_time + distance(customer, self.depo):
            return False
        dist = distance(customer, self)
        self.x = customer.x
        self.y = customer.y
        self.capacity -= customer.demand
        self.last_service_time += ceil(dist)
        self.service_route += [(customer, self.last_service_time)]
        self.last_service_time += customer.service_time
        customer.served(self.id)
        self.total_distance += dist
        if self.capacity < self.min_capacity:
            self.return_home()
        return True

    def serve_customer_force(self, customer):
        if customer.ready_time > ceil(distance(customer, self)) + self.last_service_time:
            last_service_time = self.last_service_time
            self.last_service_time = customer.ready_time - ceil(distance(customer, self))
            if self.serve_customer(customer):
                return True
            else:
                self.last_service_time = last_service_time
        return False

    def return_home(self):
        if self.x != self.depo.x or self.y != self.depo.y:
            self.capacity = self.max_capacity
            self.last_service_time += ceil(distance(self, self.depo))
            self.service_route += [(self.depo, self.last_service_time)]
            self.total_distance += distance(self, self.depo)
            self.x = self.depo.x
            self.y = self.depo.y

    def remove_customer(self, customer):
        customer_idx = [route_node[0] for route_node in self.service_route].index(customer)
        del self.service_route[customer_idx]
        self.capacity += customer.demand
        customer.unserve()

        for i, (curr_customer, curr_time) in enumerate(self.service_route[customer_idx :]):
            prev_customer, prev_time = self.service_route[customer_idx + i - 1]
            new_time = prev_time + prev_customer.service_time + ceil(distance(prev_customer, curr_customer))
            new_time = max(new_time, curr_customer.ready_time)
            self.service_route[customer_idx + i] = (curr_customer, new_time)
            
        next_customer = self.service_route[customer_idx][0]
        prev_customer = self.service_route[customer_idx - 1][0]
        self.total_distance -= (distance(customer, next_customer) + distance(customer, prev_customer))
        self.total_distance += distance(prev_customer, next_customer)
        self.last_service_time = self.service_route[-1][1]

        self.reset_vehicle_used()

    def try_to_serve_customer(self, new_customer):
        if len(self.service_route) == 1:
            return self.serve_customer(new_customer) or self.serve_customer_force(new_customer)
        shuffled = list(range(1, len(self.service_route)))
        random.shuffle(shuffled)
        for i in shuffled:
            vehicle = Vehicle(self.id, self.depo, self.max_capacity, self.min_capacity)
            vehicle.hard_reset_vehicle()
            index = i
            should_use_route = True
            new_customer.is_served = False
            for e, (customer, curr_time) in enumerate(self.service_route[1:]):
                if e + 1 == index:
                    if not vehicle.serve_customer(new_customer):
                        if not vehicle.serve_customer_force(new_customer):
                            should_use_route = False
                            break
                if not vehicle.serve_customer(customer):
                    if not vehicle.serve_customer_force(customer):
                        should_use_route = False
                        break
            if not should_use_route or not new_customer.is_served:
                new_customer.is_served = False
                continue
            self.service_route = vehicle.service_route[:]
            self.last_service_time = vehicle.last_service_time
            self.capacity = vehicle.capacity
            self.total_distance = vehicle.total_distance
            return True
        return False

    def hard_reset_vehicle(self):
        self.service_route = [(self.depo, 0)]
        self.last_service_time = 0
        self.capacity = self.max_capacity
        self.total_distance = 0

    def reset_vehicle_used(self):
        if self.service_route[0] == self.service_route[1]:
            self.hard_reset_vehicle()

    def __str__(self):
        return f'self.id = {self.id}; x={self.x}; y={self.y}; capacity={self.capacity}; last_service_time={self.last_service_time}; {self.service_route}'


def all_served(customers, b=False):
    result = True
    for c in customers:
        if not c.is_served:
            if b:
                print(c)
            result = False

    return result

class Instance:
    def __init__(self, num_vehicles, capacity, customer_list):
        assert (
            num_vehicles > 0 and capacity > 0
        ), f'Number of vehicles and their capacity must be positive! {num_vehicles}, {capacity}'
        self.num_vehicles = num_vehicles
        self.capacity = capacity

        
        depo = customer_list[0]
        self.vehicles = [Vehicle(i, depo, capacity) for i in range(num_vehicles)]
        assert (
            customer_list[0].cust_no == 0 and customer_list[0].ready_time == 0 and customer_list[0].demand == 0
        ), f'Customer list must contain depot with customer number 0!'
        self.customer_list = [customer_list[0]] + sorted(customer_list[1:], key=lambda c: c.ready_time)

    def __getitem__(self, key):
        return self.customer_list[key]

    def __str__(self):
        result = f'Vehicle Number: {self.num_vehicles}; Capacity: {self.capacity};'
        for customer in self.customer_list:
            result += f'\n{customer}'
        return result

    def sort_by_ready_time(self):
        self.customer_list.sort(key=lambda c: c.ready_time)

    def find_initial_solution(self):
        for i, v in enumerate(self.vehicles):
            while True:
                self.customer_list.sort(key = lambda c: distance(c, v) + c.ready_time)
                found = False
                for customer in self.customer_list:
                    if customer.is_served or customer.cust_no == 0:
                        continue

                    if v.serve_customer(customer):
                        found = True
                        break

                if not found:
                    for customer in self.customer_list:
                        if customer.is_served or customer.cust_no == 0:
                            continue

                        if v.serve_customer_force(customer):
                            found = True
                            break

                if not found:
                    break

            self.customer_list.sort(key = lambda c: c.cust_no)
            if all_served(self.customer_list[1:]):
                break
        self.customer_list.sort(key = lambda c: c.cust_no)

        for vehicle in self.vehicles:
            if vehicle.last_service_time == 0:
                continue
            vehicle.return_home()
        if not all_served(self.customer_list[1:], True):
            print("Not all vehicles has been served!\n")


    def generate_random_neighbour(self):
        #rand_cust = self.customer_list[random.randint(1, len(self.customer_list) - 1)]
        rand_cust = random.choices(self.customer_list[1:], [1./len(self.vehicles[c.vehicle_num].service_route) for c in self.customer_list[1:]], k = 1)[0]
        current_serving_vehicle = self.vehicles[rand_cust.vehicle_num]
        current_serving_vehicle.remove_customer(rand_cust)
        v = None
        while not rand_cust.is_served:
            if self.get_neighbour(rand_cust):
                return
            self.get_neighbour(rand_cust, True)

    def get_neighbour(self, customer, force=False):
        shuffled = [i for i in range(0, len(self.vehicles) - 1)]
        random.shuffle(shuffled)
        for vehicle_num in shuffled:
            vehicle = self.vehicles[vehicle_num]
            if force or vehicle.last_service_time != 0:
                if vehicle.try_to_serve_customer(customer):
                    return True

    def get_output(self):
        dist = 0
        result = ""
        i = 1
        for vehicle in self.vehicles:
            if vehicle.last_service_time == 0:
                continue
            vehicle.return_home()
            dist += vehicle.total_distance
            result += f'{i}: '
            for node in vehicle.service_route:
                result += f'{node[0].cust_no}({node[1]})->'
            result = result[:-2] + '\n'
            i += 1

        print("vehicle count: ", i-1)
        print("distance: ", dist)
        return f'{i-1}\n {result}{dist}\n '
    
    def get_solution(self):
        routes = []
        for vehicle in self.vehicles:
            if vehicle.last_service_time == 0:
                continue
            vehicle.return_home()
            route = []
            for node in vehicle.service_route:
                if node[0].cust_no != vehicle.depo.cust_no:
                    route.append(node[0].cust_no)
            routes.append(route)
        return routes
    
    def get_total_distance(self):
        dist = 0
        for vehicle in self.vehicles:
            if vehicle.last_service_time == 0:
                continue
            vehicle.return_home()
            dist += vehicle.total_distance
        return dist

    def get_total_distance_and_vehicles(self):
        dist = 0
        vehicles_used = 0
        for vehicle in self.vehicles:
            if vehicle.last_service_time == 0:
                continue
            vehicle.return_home()
            vehicles_used += 1
            dist += vehicle.total_distance
        return dist, vehicles_used

    def __str__(self):
        return self.get_output()

def load_from_file(filepath):
    i = 0
    customer_list = []
    num_vehicles = 0
    capacity = 0

    with open(filepath) as f:
        line = f.readline()
        while line:
            if not line.strip(): # skip empty lines
                line = f.readline()
                continue

            if i == 3: # number of vehicles and capacity
                params = [int(p) for p in line.split()]
                num_vehicles = params[0]
                capacity = params[1]
            elif i >= 6: # customer info
                params = [int(p) for p in line.split()]
                customer_list.append(Customer(*params))

            i += 1
            line = f.readline()

    return Instance(num_vehicles, capacity, customer_list)