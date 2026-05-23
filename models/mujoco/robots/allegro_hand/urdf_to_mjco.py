import mujoco, os, re

os.chdir(os.path.dirname(os.path.abspath(__file__)))

URDF_IN  = 'allegro_hand_description_right_B.urdf'
XML_OUT  = 'allegro_hand_right_B.xml'

# ── 1. Pre-process URDF ───────────────────────────────────────────────────────

content = open(URDF_IN).read()

# Remove bad inertials — MuJoCo computes from mesh geometry
content = re.sub(r'<inertial>.*?</inertial>', '', content, flags=re.DOTALL)

# Inject MuJoCo compiler hints inside <robot> tag
content = re.sub(
    r'(<robot[^>]*>)',
    r'\1\n  <mujoco><compiler discardvisual="false" autolimits="true"/></mujoco>',
    content
)

open('_allegro_fixed.urdf', 'w').write(content)

# ── 2. Convert via MuJoCo API ─────────────────────────────────────────────────

m = mujoco.MjModel.from_xml_path('_allegro_fixed.urdf')
mujoco.mj_saveLastXML(XML_OUT, m)
os.remove('_allegro_fixed.urdf')
print(f'Converted: {m.nbody} bodies, {m.njnt} joints')

# ── 3. Post-process XML ───────────────────────────────────────────────────────

xml = open(XML_OUT).read()

# Fix compiler line
xml = xml.replace(
    '<compiler angle="radian"/>',
    '<compiler angle="radian" autolimits="true"/>'
)

# Enable joint limits (autolimits not always written by mj_saveLastXML)
xml = re.sub(
    r'(<joint [^>]*range="[^"]*")',
    r'\1 limited="true"',
    xml
)

# Zero out frictionloss — placeholder values from URDF cause sluggish motion
xml = re.sub(r' frictionloss="[^"]*"', '', xml)

# Add contact exclusions for adjacent bodies
exclude_pairs = [
    ('link_0_0',  'link_1_0'),  ('link_1_0',  'link_2_0'),  ('link_2_0',  'link_3_0'),
    ('link_4_0',  'link_5_0'),  ('link_5_0',  'link_6_0'),  ('link_6_0',  'link_7_0'),
    ('link_8_0',  'link_9_0'),  ('link_9_0',  'link_10_0'), ('link_10_0', 'link_11_0'),
    ('link_12_0', 'link_13_0'), ('link_13_0', 'link_14_0'), ('link_14_0', 'link_15_0'),
]
excludes = '\n<contact>\n' + '\n'.join(
    f'  <exclude body1="{a}" body2="{b}"/>'
    for a, b in exclude_pairs
) + '\n</contact>'

xml = xml.replace('</mujoco>', excludes + '\n</mujoco>')

# Add actuators (one motor per revolute joint)
joint_names = re.findall(r'<joint name="(joint_\d+_\d+)"', xml)
actuators = '\n<actuator>\n' + '\n'.join(
    f'  <motor name="act_{j}" joint="{j}" ctrlrange="-15 15" forcerange="-15 15"/>'
    for j in joint_names
) + '\n</actuator>'

xml = xml.replace('</mujoco>', actuators + '\n</mujoco>')

open(XML_OUT, 'w').write(xml)
print(f'Post-processed: {len(joint_names)} actuators added')
print(f'Output: {XML_OUT}')