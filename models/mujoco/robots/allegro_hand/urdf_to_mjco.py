import mujoco, os, re

os.chdir(os.path.dirname(os.path.abspath(__file__)))

URDF_IN = 'allegro_hand_description_right_A.urdf'
XML_OUT = 'allegro_hand_right_A.xml'

# 1. pre-process URDF

content = open(URDF_IN).read()

# fingertip mass 0 -> 1 g (MuJoCo rejects it), zero off-diagonal inertia (source URDF has a unit error)
def _fix_inertial(m):
    block = m.group(0)
    block = re.sub(r'(<mass\s+value=")0(\.0*)?(")', r'\g<1>0.001\3', block)
    block = re.sub(r'\bixy="[^"]*"', 'ixy="0"', block)
    block = re.sub(r'\bixz="[^"]*"', 'ixz="0"', block)
    block = re.sub(r'\biyz="[^"]*"', 'iyz="0"', block)
    return block

content = re.sub(r'<inertial>.*?</inertial>', _fix_inertial, content, flags=re.DOTALL)

# fusestatic="false" keeps palm_link as a named body instead of merging it into worldbody
content = re.sub(
    r'(<robot[^>]*>)',
    r'\1\n  <mujoco><compiler discardvisual="false" autolimits="true" fusestatic="false"/></mujoco>',
    content
)

open('_allegro_fixed.urdf', 'w').write(content)

# 2. convert via MuJoCo API

m = mujoco.MjModel.from_xml_path('_allegro_fixed.urdf')
mujoco.mj_saveLastXML(XML_OUT, m)
try:
    os.remove('_allegro_fixed.urdf')
except Exception:
    pass
print('Converted: {} bodies, {} joints'.format(m.nbody, m.njnt))

# 3. post-process XML

xml = open(XML_OUT).read()

xml = re.sub(
    r'<compiler[^/]*/?>',
    '<compiler angle="radian" meshdir="assets" autolimits="true" fusestatic="false"/>',
    xml
)

xml = re.sub(r'file="assets/([^"]+)"', r'file="\1"', xml)

xml = re.sub(
    r'(<joint [^>]*range="[^"]*")',
    r'\1 limited="true"',
    xml
)

# drop frictionloss, URDF placeholder values make motion sluggish
xml = re.sub(r' frictionloss="[^"]*"', '', xml)

body_names = re.findall(r'<body name="([^"]+)"', xml)
print('Bodies found: {}'.format(body_names))

exclude_pairs = [
    ('link_0_0',  'link_1_0'),  ('link_1_0',  'link_2_0'),  ('link_2_0',  'link_3_0'),
    ('link_4_0',  'link_5_0'),  ('link_5_0',  'link_6_0'),  ('link_6_0',  'link_7_0'),
    ('link_8_0',  'link_9_0'),  ('link_9_0',  'link_10_0'), ('link_10_0', 'link_11_0'),
    ('link_12_0', 'link_13_0'), ('link_13_0', 'link_14_0'), ('link_14_0', 'link_15_0'),
]
exclude_pairs = [(a, b) for a, b in exclude_pairs if a in body_names and b in body_names]

excludes = '\n<contact>\n' + '\n'.join(
    '  <exclude body1="{}" body2="{}"/>'.format(a, b)
    for a, b in exclude_pairs
) + '\n</contact>'

xml = xml.replace('</mujoco>', excludes + '\n</mujoco>')

joint_names = re.findall(r'<joint name="(joint_\d+_\d+)"', xml)
actuators = '\n<actuator>\n' + '\n'.join(
    '  <motor name="act_{}" joint="{}" ctrlrange="-15 15" forcerange="-15 15"/>'.format(j, j)
    for j in joint_names
) + '\n</actuator>'

xml = xml.replace('</mujoco>', actuators + '\n</mujoco>')

open(XML_OUT, 'w').write(xml)
print('Post-processed: {} actuators, {} contact excludes'.format(len(joint_names), len(exclude_pairs)))
print('Output: {}'.format(XML_OUT))
