# from params_proto import PrefixProto
class URDFCfg:
    root_folder_path="locotouch/utils/urdf_processor/go1/urdf/"
    row_num=17
    column_num=13
    row_distance = 0.0143
    column_distance = 0.0128
    first_sensor_position = [0.1144, 0.0768, 0.093]
    tmp_original_leg_definition = ["1_FR", "2_FL", "3_RR", "4_RL"]
    tmp_new_leg_definition = ["a_FR", "b_FL", "c_RR", "d_RL"]

    tmp_urdf_path="locotouch_template.urdf"
    locotouch_urdf_path="locotouch.urdf"

class URDFCfgCollapse(URDFCfg):
    # tactile sensors are collapsed for policies that do not use tactile inputs
    tmp_urdf_path="locotouch_without_tactile_template.urdf"
    locotouch_urdf_path="locotouch_without_tactile.urdf"


class URDFGenerator:
    def __init__(self, cfg: URDFCfg):
        self.cfg = cfg

        self.tmp_folder_path = self.cfg.root_folder_path + self.cfg.tmp_urdf_path
        with open(self.tmp_folder_path, "r") as f:
            self.tmp_urdf = f.read()

        self.locotouch_urdf_path = self.cfg.root_folder_path + self.cfg.locotouch_urdf_path

        self.row_num = self.cfg.row_num
        self.column_num = self.cfg.column_num
        self.sensor_num = self.row_num * self.column_num
        self.row_distance = self.cfg.row_distance
        self.column_distance = self.cfg.column_distance
        self.first_sensor_position = self.cfg.first_sensor_position

        self.tmp_original_leg_definition = self.cfg.tmp_original_leg_definition
        self.tmp_new_leg_definition = self.cfg.tmp_new_leg_definition


    def generate_locotouch_urdf(self):
        # replace original leg definition with new leg definition
        for i in range(len(self.tmp_original_leg_definition)):
            self.tmp_urdf = self.tmp_urdf.replace(self.tmp_original_leg_definition[i], self.tmp_new_leg_definition[i])

        # extract the sensor, joint, and suffix parts of the urdf
        sensor_start_idx = self.tmp_urdf.find("<link name=\"sensor_01_01\"")
        joint_start_idx = self.tmp_urdf.find("<joint name=\"sensor_01_01_fixed\"")
        suffix_start_idx = self.tmp_urdf.find("</robot>")
        self.sensor_template = self.tmp_urdf[sensor_start_idx:joint_start_idx]
        self.joint_template = self.tmp_urdf[joint_start_idx:suffix_start_idx]
        self.urdf_suffix_part = self.tmp_urdf[suffix_start_idx:]
        self.tmp_urdf = self.tmp_urdf[:sensor_start_idx]

        # save the sensors and joints for all rows
        all_sensors = []
        all_joints = []
        for row in range(1, self.row_num+1):
            for col in range(1, self.column_num+1):
                row_id = str(row) if row >= 10 else "0" + str(row)
                col_id = str(col) if col >= 10 else "0" + str(col)
                sensor_name = f"sensor_{row_id}_{col_id}"
                joint_name = f"sensor_{row_id}_{col_id}_fixed"
                
                sensor = self.sensor_template.replace("sensor_01_01", sensor_name)
                joint = self.joint_template.replace("sensor_01_01_fixed", joint_name)
                joint = joint.replace("sensor_01_01", sensor_name)
                x_origin = round(self.first_sensor_position[0] - self.row_distance * (row - 1), 4)
                y_origin = round(self.first_sensor_position[1] - self.column_distance * (col - 1), 4)
                z_origin = round(self.first_sensor_position[2], 4)
                joint = joint.replace("origin xyz=\"0 0 0\"", f"origin xyz=\"{x_origin} {y_origin} {z_origin}\"")

                all_sensors.append(sensor)
                all_joints.append(joint)
        
        for sensor, joint in zip(all_sensors, all_joints):
            self.tmp_urdf += sensor + "\n"
            self.tmp_urdf += joint + "\n"
        
        self.tmp_urdf += self.urdf_suffix_part
        
        with open(self.locotouch_urdf_path, "w") as f:
            f.write(self.tmp_urdf)





if __name__ == "__main__":
    cfg = URDFCfg()
    generator = URDFGenerator(cfg)
    generator.generate_locotouch_urdf()

    cfg_collapse = URDFCfgCollapse()
    generator_collapse = URDFGenerator(cfg_collapse)
    generator_collapse.generate_locotouch_urdf()





"""
python locotouch/utils/urdf_processor/generate_locotouch_urdf.py

python ../IsaacLab/scripts/tools/convert_urdf.py locotouch/utils/urdf_processor/go1/urdf/locotouch.urdf locotouch/assets/locotouch/locotouch_instanceable_new1.usd --merge-joints --make-instanceable
python ../IsaacLab/scripts/tools/convert_urdf.py locotouch/utils/urdf_processor/go1/urdf/locotouch.urdf locotouch/assets/locotouch/locotouch_new.usd --merge-joints
python ../IsaacLab/scripts/tools/convert_urdf.py locotouch/utils/urdf_processor/go1/urdf/locotouch_without_tactile.urdf locotouch/assets/locotouch/locotouch_without_tactile_instanceable_new.usd --merge-joints --make-instanceable
python ../IsaacLab/scripts/tools/convert_urdf.py locotouch/utils/urdf_processor/go1/urdf/locotouch_without_tactile.urdf locotouch/assets/locotouch/locotouch_without_tactile_new.usd --merge-joints

"""

