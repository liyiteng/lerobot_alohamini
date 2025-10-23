#### View all motor states
```
python examples/debug/motors.py get_motors_states \
  --port /dev/ttyACM0
```
#### Control the mobile base only
```
python examples/debug/wheels.py \
   --port /dev/ttyACM0
```

#### Control the lift axis only
```
python examples/debug/axis.py \
   --port /dev/ttyACM0
```

#### Disable torque for all arm motors
```
python examples/debug/motors.py reset_motors_torque  \
  --port /dev/ttyACM0
```

#### Rotate a specific motor by ID
```
python examples/debug/motors.py move_motor_to_position \
  --id 1 \
  --position 2 \
  --port /dev/ttyACM1
```


#### Set a new motor ID
```
python examples/debug/motors.py configure_motor_id \
  --id 10 \
  --set_id 8 \
  --port /dev/ttyACM0
```


#### Reset current position as the motor midpoint
```
python examples/debug/motors.py reset_motors_to_midpoint \
  --port /dev/ttyACM1
```


#### Execute an action script on the robot arm
```
python examples/debug/motors.py move_motors_by_script \
   --script_path action_scripts/test_dance.txt  \
   --port /dev/ttyACM0
```

