# Copyright 2010-2022 Google LLC
"""This model implements a bus driver scheduling problem.

Constraints:
- max driving time per driver <= 9h
- max working time per driver <= 12h
- min working time per driver >= 6.5h (soft)
- 30 min break after each 4h of driving time per driver
- 10 min preparation time before the first shift
- 15 min cleaning time after the last shift
- 2 min waiting time after each shift for passenger boarding and alighting
"""

import collections
import math
from data import SAMPLE_SHIFTS_SMALL, SAMPLE_SHIFTS_MEDIUM, SAMPLE_SHIFTS_LARGE

from absl import app
from absl import flags
from google.protobuf import text_format
from ortools.sat.python import cp_model

_OUTPUT_PROTO = flags.DEFINE_string(
    'output_proto', '', 'Output file to write the cp_model proto to.')
_PARAMS = flags.DEFINE_string('params',
                              'num_search_workers:20,log_search_progress:true',
                              'Sat solver parameters.')
_INSTANCE = flags.DEFINE_integer('instance', 2, 'Instance to select (1, 2, 3).',
                                 1, 3)

def bus_driver_scheduling(minimize_drivers, max_num_drivers):
    """Optimize the bus driver scheduling problem.

  This model has two modes.

  If minimize_drivers == True, the objective will be to find the minimal
  number of drivers, independently of the working times of each drivers.

  Otherwise, will will create max_num_drivers non optional drivers, and
  minimize the sum of working times of these drivers.

  Args:
    minimize_drivers: A Boolean parameter specifying the objective of the
      problem. If True, it tries to minimize the number of used drivers. If
      false, it minimizes the sum of working times per workers.
    max_num_drivers: This number specifies the exact number of non optional
      drivers to use. This is only used if 'minimize_drivers' is False.

  Returns:
      The objective value of the model.
  """
    shifts = None
    if _INSTANCE.value == 1:
        shifts = SAMPLE_SHIFTS_SMALL
    elif _INSTANCE.value == 2:
        shifts = SAMPLE_SHIFTS_MEDIUM
    elif _INSTANCE.value == 3:
        shifts = SAMPLE_SHIFTS_LARGE

    num_shifts = len(shifts)

    # All durations are in minutes.
    max_driving_time = 540  # 9 hours.
    max_driving_time_without_pauses = 240  # 4 hours
    min_pause_after_4h = 30
    min_delay_between_shifts = 2
    max_working_time = 720  # 12 hours
    min_working_time = 390  # 6.5 hours
    setup_time = 10
    cleanup_time = 15

    # Computed data.
    total_driving_time = sum(shift[5] for shift in shifts)
    min_num_drivers = int(math.ceil(total_driving_time * 1.0 /
                                    max_driving_time))
    num_drivers = 2 * min_num_drivers if minimize_drivers else max_num_drivers
    min_start_time = min(shift[3] for shift in shifts)
    max_end_time = max(shift[4] for shift in shifts)

    print('Bus driver scheduling')
    print('  num shifts =', num_shifts)
    print('  total driving time =', total_driving_time, 'minutes')
    print('  min num drivers =', min_num_drivers)
    print('  num drivers =', num_drivers)
    print('  min start time =', min_start_time)
    print('  max end time =', max_end_time)

    model = cp_model.CpModel()

    # For each driver and each shift, we store:
    #   - the total driving time including this shift
    #   - the acrued driving time since the last 30 minute break
    # Special arcs have the following effect:
    #   - 'from source to shift' sets the starting time and accumulate the first
    #      shift
    #   - 'from shift to end' sets the ending time, and fill the driving_times
    #      variable
    # Arcs between two shifts have the following impact
    #   - add the duration of the shift to the total driving time
    #   - reset the accumulated driving time if the distance between the two
    #     shifts is more than 30 minutes, add the duration of the shift to the
    #     accumulated driving time since the last break otherwise

    # Per (driver, node) info (driving time, active,
    # driving time since break)
    total_driving = {}
    no_break_driving = {}
    active = {}
    starting_shifts = {}

    # Per driver info (start, end, driving times, is working)
    start_times = []
    end_times = []
    driving_times = []
    working_drivers = []
    working_times = []

    # Weighted objective
    delay_literals = []
    delay_weights = []

    # Used to propagate more between drivers
    shared_incoming_shift = collections.defaultdict(list)
    shared_outgoing_literals = collections.defaultdict(list)

    for d in range(num_drivers):
        start_times.append(
            model.NewIntVar(min_start_time - setup_time, max_end_time,
                            'start_%i' % d))
        end_times.append(
            model.NewIntVar(min_start_time, max_end_time + cleanup_time,
                            'end_%i' % d))
        driving_times.append(
            model.NewIntVar(0, max_driving_time, 'driving_%i' % d))
        working_times.append(
            model.NewIntVar(0, max_working_time, 'working_times_%i' % d))

        incoming_shift = collections.defaultdict(list)
        outgoing_shift = collections.defaultdict(list)
        outgoing_source_shift = []
        incoming_sink_literals = []

        # Create all the shift variables before iterating on the transitions
        # between these shifts.
        for s in range(num_shifts):
            total_driving[d, s] = model.NewIntVar(0, max_driving_time,
                                                  'dr_%i_%i' % (d, s))
            no_break_driving[d, s] = model.NewIntVar(
                0, max_driving_time_without_pauses, 'mdr_%i_%i' % (d, s))
            active[d, s] = model.NewBoolVar('performed_%i_%i' % (d, s))

        for s in range(num_shifts):
            shift = shifts[s]
            duration = shift[5]

            # Arc from source to shift.
            #    - set the start time of the driver
            #    - increase driving time and driving time since break
            source_shift = model.NewBoolVar('%i from source to %i' % (d, s))
            outgoing_source_shift.append(source_shift)
            incoming_shift[s].append(source_shift)
            shared_incoming_shift[s].append(source_shift)
            model.Add(start_times[d] == shift[3] -
                      setup_time).OnlyEnforceIf(source_shift)
            model.Add(total_driving[d, s] == duration).OnlyEnforceIf(source_shift)
            model.Add(no_break_driving[d,
                                       s] == duration).OnlyEnforceIf(source_shift)
            starting_shifts[d, s] = source_shift

            # Arc from shift to end: the last shift of the driver
            #    - set the end working time of the driver
            #    - set the driving times of the driver
            final_shift = model.NewBoolVar('%i from %i to sink' % (d, s))
            outgoing_shift[s].append(final_shift)
            shared_outgoing_literals[s].append(final_shift)
            incoming_sink_literals.append(final_shift)
            model.Add(end_times[d] == shift[4] +
                      cleanup_time).OnlyEnforceIf(final_shift)
            model.Add(
                driving_times[d] == total_driving[d, s]).OnlyEnforceIf(final_shift)

            # Node not active
            #    - set both driving times to 0
            #    - add a looping arc on the node
            model.Add(total_driving[d,
                                    s] == 0).OnlyEnforceIf(active[d,
                                                                     s].Not())
            model.Add(no_break_driving[d, s] == 0).OnlyEnforceIf(
                active[d, s].Not())
            incoming_shift[s].append(active[d, s].Not())
            outgoing_shift[s].append(active[d, s].Not())
            # Not adding to the shared lists, because, globally, each node will have
            # one incoming shift, and one outgoing shift.

            # Node active:
            #    - add upper bound on start_time
            #    - add lower bound on end_times
            model.Add(start_times[d] <= shift[3] - setup_time).OnlyEnforceIf(
                active[d, s])
            model.Add(end_times[d] >= shift[4] + cleanup_time).OnlyEnforceIf(
                active[d, s])

            for o in range(num_shifts):
                other = shifts[o]
                delay = other[3] - shift[4]
                if delay < min_delay_between_shifts:
                    continue
                arc = model.NewBoolVar('%i from %i to %i' % (d, s, o))

                # Increase driving time
                model.Add(total_driving[d, o] == total_driving[d, s] +
                          other[5]).OnlyEnforceIf(arc)

                # Increase no_break_driving or reset it to 0 depending on the delay
                if delay >= min_pause_after_4h:
                    model.Add(
                        no_break_driving[d, o] == other[5]).OnlyEnforceIf(arc)
                else:
                    model.Add(no_break_driving[d, o] == no_break_driving[d, s] +
                              other[5]).OnlyEnforceIf(arc)

                # Add arc
                outgoing_shift[s].append(arc)
                shared_outgoing_literals[s].append(arc)
                incoming_shift[o].append(arc)
                shared_incoming_shift[o].append(arc)

                # Cost part
                delay_literals.append(arc)
                delay_weights.append(delay)

        model.Add(working_times[d] == end_times[d] - start_times[d])

        if minimize_drivers:
            # Driver is not working.
            working = model.NewBoolVar('working_%i' % d)
            model.Add(start_times[d] == min_start_time).OnlyEnforceIf(
                working.Not())
            model.Add(end_times[d] == min_start_time).OnlyEnforceIf(
                working.Not())
            model.Add(driving_times[d] == 0).OnlyEnforceIf(working.Not())
            working_drivers.append(working)
            outgoing_source_shift.append(working.Not())
            incoming_sink_literals.append(working.Not())

            # Conditional working time constraints
            model.Add(
                working_times[d] >= min_working_time).OnlyEnforceIf(working)
            model.Add(working_times[d] == 0).OnlyEnforceIf(working.Not())
        else:
            # Working time constraints
            model.Add(working_times[d] >= min_working_time)

        # Create circuit constraint.
        model.AddExactlyOne(outgoing_source_shift)
        for s in range(num_shifts):
            model.AddExactlyOne(outgoing_shift[s])
            model.AddExactlyOne(incoming_shift[s])
        model.AddExactlyOne(incoming_sink_literals)

    # Each shift is covered.
    for s in range(num_shifts):
        model.AddExactlyOne(active[d, s] for d in range(num_drivers))
        # Globally, each node has one incoming and one outgoing literal
        model.AddExactlyOne(shared_incoming_shift[s])
        model.AddExactlyOne(shared_outgoing_literals[s])

    # Symmetry breaking

    # The first 3 shifts must be active by 3 different drivers.
    # Let's assign them to the first 3 drivers in sequence
    model.Add(starting_shifts[0, 0] == 1)
    model.Add(starting_shifts[1, 1] == 1)
    model.Add(starting_shifts[2, 2] == 1)

    if minimize_drivers:
        # Push non working drivers to the end
        for d in range(num_drivers - 1):
            model.AddImplication(working_drivers[d].Not(),
                                 working_drivers[d + 1].Not())

    # Redundant constraints: sum of driving times = sum of shift driving times
    model.Add(cp_model.LinearExpr.Sum(driving_times) == total_driving_time)
    if not minimize_drivers:
        model.Add(
            cp_model.LinearExpr.Sum(working_times) == total_driving_time +
            num_drivers * (setup_time + cleanup_time) +
            cp_model.LinearExpr.WeightedSum(delay_literals, delay_weights))

    if minimize_drivers:
        # Minimize the number of working drivers
        model.Minimize(cp_model.LinearExpr.Sum(working_drivers))
    else:
        # Minimize the sum of delays between tasks, which in turns minimize the
        # sum of working times as the total driving time is fixed
        model.Minimize(
            cp_model.LinearExpr.WeightedSum(delay_literals, delay_weights))

    if not minimize_drivers and _OUTPUT_PROTO.value:
        print('Writing proto to %s' % _OUTPUT_PROTO.value)
        with open(_OUTPUT_PROTO.value, 'w') as text_file:
            text_file.write(str(model))

    # Solve model.
    solver = cp_model.CpSolver()
    if _PARAMS.value:
        text_format.Parse(_PARAMS.value, solver.parameters)

    status = solver.Solve(model)

    if status != cp_model.OPTIMAL and status != cp_model.FEASIBLE:
        return -1

    # Display solution
    if minimize_drivers:
        max_num_drivers = int(solver.ObjectiveValue())
        print('minimal number of drivers =', max_num_drivers)
        return max_num_drivers

    for d in range(num_drivers):
        print('Driver %i: ' % (d + 1))
        print('  total driving time =', solver.Value(driving_times[d]))
        print('  working time =',
              solver.Value(working_times[d]) + setup_time + cleanup_time)

        first = True
        for s in range(num_shifts):
            shift = shifts[s]

            if not solver.BooleanValue(active[d, s]):
                continue

            # If the interval between the termination of the preceding shift and the commencement of
            # the current shift exceeds 30 minutes, we examine the uninterrupted driving that has been restarted.
            if solver.Value(no_break_driving[d, s]) == shift[5] and not first:
                print('    **break**')
            print('    shift ', shift[0], ':', shift[1], '-', shift[2])
            first = False

    return int(solver.ObjectiveValue())


def main(_):
    """Optimize the bus driver allocation in two phases."""
    print('----------- first phase pass: minimize the number of drivers')

    num_drivers = bus_driver_scheduling(True, -1)
    if num_drivers == -1:
        print('no solution found, skipping the final phase')
    else:
        print('----------- second pass: minimize the sum of working times')
        bus_driver_scheduling(False, num_drivers)


if __name__ == '__main__':
    app.run(main)