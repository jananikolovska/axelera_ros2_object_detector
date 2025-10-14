cd ~/voyager-sdk
source venv/bin/activate
source /opt/ros/humble/setup.bash
cd ros2_ws
source install/setup.bash
export PYTHONPATH=$VIRTUAL_ENV/lib/python3.10/site-packages:$PYTHONPATH
