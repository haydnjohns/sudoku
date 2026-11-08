"""Low-level control of the left and right stepper motors."""

from threading import Thread
from time import sleep
from math import pi
from gpiozero import DigitalOutputDevice


LEFT_STEPPER_GPIO = (5, 6, 13, 19)
RIGHT_STEPPER_GPIO = (12, 16, 20, 21)
INCREMENTS_PER_REVOLUTION = 4096
STEP_DELAY = 0.0015
WHEEL_DIAMETER_MM = 65.0
WHEEL_BASE_MM = 150.0
STEP_SEQUENCE = [
        [1, 0, 0, 0],
        [1, 1, 0, 0],
        [0, 1, 0, 0],
        [0, 1, 1, 0],
        [0, 0, 1, 0],
        [0, 0, 1, 1],
        [0, 0, 0, 1],
        [1, 0, 0, 1]
    ]
left_stepper_pins = None
right_stepper_pins = None


def initialise_steppers():
    """Create and return the GPIO outputs used by both stepper motors."""
    global left_stepper_pins, right_stepper_pins

    left_stepper_pins = [
        DigitalOutputDevice(pin, initial_value=False)
        for pin in LEFT_STEPPER_GPIO
    ]
    right_stepper_pins = [
        DigitalOutputDevice(pin, initial_value=False)
        for pin in RIGHT_STEPPER_GPIO
    ]

    return (left_stepper_pins, right_stepper_pins, STEP_SEQUENCE, INCREMENTS_PER_REVOLUTION)


def set_increment(pins, increment):
    """Set one stepper motor's coils to a step in the sequence."""
    for pin, value in zip(pins, increment):
        pin.value = value


def move_stepper(pins, revolutions):
    """Move one stepper motor by a number of revolutions."""
    sequence = STEP_SEQUENCE if revolutions >= 0 else STEP_SEQUENCE[::-1]
    increments = round(abs(revolutions) * INCREMENTS_PER_REVOLUTION)
    for step in range(increments):
        set_increment(pins, sequence[step % len(sequence)])
        sleep(STEP_DELAY)


def move_both_steppers(left_revolutions, right_revolutions):
    """Move both stepper motors concurrently."""
    if left_stepper_pins is None or right_stepper_pins is None:
        initialise_steppers()

    left_thread = Thread(target=move_stepper, args=(left_stepper_pins, left_revolutions))
    right_thread = Thread(target=move_stepper, args=(right_stepper_pins, right_revolutions))

    left_thread.start()
    right_thread.start()
    left_thread.join()
    right_thread.join()


def move_straight(distance):
    """Move both wheels by *distance* millimetres."""
    revolutions = distance / (pi * WHEEL_DIAMETER_MM)
    move_both_steppers(revolutions, revolutions)


def pivot(pivot_point, angle):
    """Pivot by *angle* degrees around the left wheel, right wheel, or centre."""
    angle_radians = angle * pi / 180
    wheel_circumference = pi * WHEEL_DIAMETER_MM

    if pivot_point == "left_wheel":
        left_revolutions = 0
        right_revolutions = angle_radians * WHEEL_BASE_MM / wheel_circumference
    elif pivot_point == "right_wheel":
        left_revolutions = -angle_radians * WHEEL_BASE_MM / wheel_circumference
        right_revolutions = 0
    else:
        revolutions = (angle_radians * (WHEEL_BASE_MM / 2) / wheel_circumference)
        left_revolutions = -revolutions
        right_revolutions = revolutions

    move_both_steppers(left_revolutions, right_revolutions)


def control_steppers(move_sequence):
    """Execute movement commands as (left revs, right revs)."""
    for move in move_sequence:
        move_both_steppers(*move)


# Demo
 
if __name__ == "__main__":

    DEMO_DISTANCE_MM = 100.0
    DEMO_ANGLE = 30
    DEMO_WAIT = 0.5

    # print("Driving forward")
    # move_straight(DEMO_DISTANCE_MM)
    # sleep(DEMO_WAIT)

    # print("Driving backward")
    # move_straight(-DEMO_DISTANCE_MM)
    # sleep(DEMO_WAIT)

    # print("Pivoting on the left wheel")
    # pivot("left_wheel", DEMO_ANGLE)
    # sleep(DEMO_WAIT)

    # print("Pivoting on the right wheel")
    # pivot("right_wheel", DEMO_ANGLE)
    # sleep(DEMO_WAIT)

    print("CCW Pivot")
    pivot("centre", DEMO_ANGLE)
    sleep(DEMO_WAIT)

    print("CW Pivot")
    pivot("centre", DEMO_ANGLE)
    sleep(DEMO_WAIT)
