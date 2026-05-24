import mujoco, os, re

os.chdir(os.path.dirname(os.path.abspath(__file__)))

URDF_IN = 'allegro_hand_description_right_A.urdf'
XML_OUT = 'allegro_hand_right_A.xml'

# -- 1. Pre-process URDF ------------------------------------------------------

content = open(URDF_IN).read()

# Keep the source URDF's world -> palm_link structure intact.
# We need palm_link to be a proper named body in the MuJoCo XML so that its
# inertia is preserved and the scene builder can re-parent it under fr3_link7.
# MuJoCo's default fusestatic=true would merge palm_link (fixed joint to world)
# into worldbody, discarding its name and inertia.  Setting fusestatic="false"
# prevents that and keeps palm_link as a named child of worldbody.
#
# Note: fingertip fixed joints (link_X_0_tip) are also kept unfused; they
# appear as lightweight separate bodies -- harmless for gravity compensation.

# Keep real hardware inertials so MuJoCo's mass distribution matches the
# Pinocchio URDF used for gravity compensation.  Two corrections are applied:
#
# 1. Zero-mass fingertip links: the source URDF marks five fingertip links with
#    mass="0.00", which MuJoCo rejects for non-root bodies.  Replace with 1 g.
#
# 2. Corrupted off-diagonal inertia: several links (link_2_0, link_3_0 and the
#    equivalent finger-3/7/11 links) have ixy values ~5e-3, which is ~1000x
#    larger than their diagonal terms -- clearly a unit error in the source URDF.
#    A non-positive-definite tensor causes MuJoCo to abort.  Zero all
#    off-diagonal terms; the diagonal ixx/iyy/izz values are correct and that
#    is all that matters for gravity-compensation accuracy.
def _fix_inertial(m):
    block = m.group(0)
    # Fix zero mass
    block = re.sub(r'(<mass\s+value=")0(\.0*)?(")', r'\g<1>0.001\3', block)
    # Zero off-diagonal inertia terms
    block = re.sub(r'\bixy="[^"]*"', 'ixy="0"', block)
    block = re.sub(r'\bixz="[^"]*"', 'ixz="0"', block)
    block = re.sub(r'\biyz="[^"]*"', 'iyz="0"', block)
    return block

content = re.sub(r'<inertial>.*?</inertial>', _fix_inertial, content, flags=re.DOTALL)

# Inject MuJoCo compiler hints inside <robot> tag.
# fusestatic="false" is the critical flag: it prevents palm_link from being
# merged into worldbody despite its fixed joint to the "world" root link.
content = re.sub(
    r'(<robot[^>]*>)',
    r'\1\n  <mujoco><compiler discardvisual="false" autolimits="true" fusestatic="false"/></mujoco>',
    content
)

open('_allegro_fixed.urdf', 'w').write(content)

# -- 2. Convert via MuJoCo API ------------------------------------------------

m = mujoco.MjModel.from_xml_path('_allegro_fixed.urdf')
mujoco.mj_saveLastXML(XML_OUT, m)
try:
    os.remove('_allegro_fixed.urdf')
except Exception:
    pass
print('Converted: {} bodies, {} joints'.format(m.nbody, m.njnt))

# -- 3. Post-process XML ------------------------------------------------------

xml = open(XML_OUT).read()

# Fix compiler -- add meshdir="assets" and autolimits, strip double assets/ from file paths
xml = re.sub(
    r'<compiler[^/]*/?>',
    '<compiler angle="radian" meshdir="assets" autolimits="true" fusestatic="false"/>',
    xml
)

# Strip "assets/" prefix from mesh file attributes -- meshdir already handles the folder
xml = re.sub(r'file="assets/([^"]+)"', r'file="\1"', xml)

# Enable joint limits (autolimits not always written by mj_saveLastXML)
xml = re.sub(
    r'(<joint [^>]*range="[^"]*")',
    r'\1 limited="true"',
    xml
)

# Zero out frictionloss -- URDF placeholder values cause sluggish motion
xml = re.sub(r' frictionloss="[^"]*"', '', xml)

# Get actual body names for contact excludes
body_names = re.findall(r'<body name="([^"]+)"', xml)
print('Bodies found: {}'.format(body_names))

# Build adjacent exclude pairs from kinematic chain
# Fingers 0-2: link_X_0 chains; Thumb: link_12-15
exclude_pairs = [
    ('link_0_0',  'link_1_0'),  ('link_1_0',  'link_2_0'),  ('link_2_0',  'link_3_0'),
    ('link_4_0',  'link_5_0'),  ('link_5_0',  'link_6_0'),  ('link_6_0',  'link_7_0'),
    ('link_8_0',  'link_9_0'),  ('link_9_0',  'link_10_0'), ('link_10_0', 'link_11_0'),
    ('link_12_0', 'link_13_0'), ('link_13_0', 'link_14_0'), ('link_14_0', 'link_15_0'),
]
# Only include pairs where both bodies actually exist in the XML
exclude_pairs = [(a, b) for a, b in exclude_pairs if a in body_names and b in body_names]

excludes = '\n<contact>\n' + '\n'.join(
    '  <exclude body1="{}" body2="{}"/>'.format(a, b)
    for a, b in exclude_pairs
) + '\n</contact>'

xml = xml.replace('</mujoco>', excludes + '\n</mujoco>')

# Add actuators -- one motor per revolute joint
joint_names = re.findall(r'<joint name="(joint_\d+_\d+)"', xml)
actuators = '\n<actuator>\n' + '\n'.join(
    '  <motor name="act_{}" joint="{}" ctrlrange="-15 15" forcerange="-15 15"/>'.format(j, j)
    for j in joint_names
) + '\n</actuator>'

xml = xml.replace('</mujoco>', actuators + '\n</mujoco>')

open(XML_OUT, 'w').write(xml)
print('Post-processed: {} actuators, {} contact excludes'.format(len(joint_names), len(exclude_pairs)))
print('Output: {}'.format(XML_OUT))
