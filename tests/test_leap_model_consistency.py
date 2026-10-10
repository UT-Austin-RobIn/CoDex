"""Saved CoDex grasps require identical joint frames in solver and simulation."""
from pathlib import Path
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1] / 'assets/models/leap_rh'


def test_joint_frames_match_solver():
    solver = {j.attrib['name']: j for j in ET.parse(ROOT/'leap_solver.urdf').getroot().findall('joint')}
    for filename in ['leap_sim.urdf', 'floating_leap.urdf']:
        sim = {j.attrib['name']: j for j in ET.parse(ROOT/filename).getroot().findall('joint')}
        for name, joint in solver.items():
            assert name in sim, (filename, name)
            for tag in ['origin', 'axis', 'parent', 'child', 'limit']:
                a, b = joint.find(tag), sim[name].find(tag)
                assert (None if a is None else a.attrib) == (None if b is None else b.attrib), (filename, name, tag)


if __name__ == '__main__':
    test_joint_frames_match_solver()
    print('LEAP solver/simulation joint-frame contract passed')
