viewpoint_template = "Viewpoint {p}:\n"
direction_template = "Robot {action} and reach viewpoint {p}.\n"

remaining_prompt = \
"The {view_num} images provided, arranged in sequential order, visualize a robot navigation trajectory in robot first view.\n"+\
"For every two adjacent viewpoints, the direction of travel from the previous viewpoint to the next viewpoint has been provided above in the form of text, "+\
"and the direction of travel to the next viewpoint is also indicated by a red arrow on the image, which is consistent with the text description.\n"+\
"Your task is to describe this trajectory in the form of instructions based on the images and the direction of each step provided to you earlier.\n"+\
"Your description can include details about landmark objects, room descriptions, direction etc.\n"+\
"The principle of description is to express the robot's walking position as clearly as possible, so that other intelligent agents can also reach the end position.\n"+\
"Please describe and return your instructions in the following format\n"+\
"Instruction: {{instruction}}\n"+\
"In addition, please follow the following principles when generating answers:\n"+\
"1. Don't use cardinal directions;\n"+\
"2. Carefully describe the relative orientation of objects and robot in the scene, especially distinguishing between left and right, as this may be misleading. "+\
"If you are unsure whether the object is on the left or right side of the robot, you can use some other expressions instead, such as \"beside\" or \"nearby\" "+\
"(you can choose the appropriate expression yourself);\n"+\
"3. The instruction consists of approximately {view_num} * 5 words, and \n"+\
"4. Do not use the markdown format to return the result, just return the text result according to the format provided in the previous text;\n"+\
"5. You need to carefully observe the red arrows in each viewpoint image, as they indicate the direction of next viewpoint. (never mention the red arrow in the answer, it does not belong to the environment itself, "+\
"but is only used to assist you in describing it).\n"+\
"6. You need to understand that because the entire trajectory is continuous, there may be viewpoint overlap between the preceding and following frames. Therefore, for example, if the robot "+\
"trajectory passes through a door, you may see that door in multiple viewpoints, which does not mean that the robot has passed through multiple doors. Therefore, before giving your "+\
"description, you need to be aware of which objects in the adjacent viewpoints are the same, in order to avoid describing them as multiple objects in the instruction."