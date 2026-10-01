/* Anatomy configuration: which parts of thorax.glb are drawn, how, and what they mean.
 *
 * Node names inside the GLB look like  group|BodyParts3D-id|Part_name  (see scripts/build_thorax_glb.py).
 * To add a structure, add its group to GROUPS (and to REGIONS if it should be selectable).
 * To add a predicted vessel, add it to VESSEL_GROUPS and to the API's TARGETS.
 */
export const MODEL_URL = '/models/thorax.glb';

export const VESSELS = ['LAD', 'LCX', 'RCA'];
export const VESSEL_GROUPS = { lad: 'LAD', lcx: 'LCX', rca: 'RCA' };   // glb group -> predicted target
export const NEUTRAL_VESSEL_GROUPS = ['lm'];                           // left main trunk: not a prediction target

export const VESSEL_INFO = {
  LAD: 'Left anterior descending: supplies the front of the heart',
  LCX: 'Left circumflex: supplies the side and back of the heart',
  RCA: 'Right coronary artery: supplies the right side and bottom of the heart',
};

/* kind: 'skeleton' (translucent, never pickable) | 'heart' (opaque, selectable region) */
export const GROUPS = {
  rib: { kind: 'skeleton', color: 0xe6dcc6, opacity: 0.17 },
  cart: { kind: 'skeleton', color: 0xd3e6ee, opacity: 0.2 },
  sternum: { kind: 'skeleton', color: 0xe6dcc6, opacity: 0.14 },
  clav: { kind: 'skeleton', color: 0xe6dcc6, opacity: 0.16 },
  vert: { kind: 'skeleton', color: 0xe6dcc6, opacity: 0.18 },
  disc: { kind: 'skeleton', color: 0xc9d7de, opacity: 0.18 },
  vent: { kind: 'heart', color: 0xb85c5c },
  atria: { kind: 'heart', color: 0xc9726f },
  aorta: { kind: 'heart', color: 0xd4756b },
  pulm: { kind: 'heart', color: 0x6a86c0 },
  pvein: { kind: 'heart', color: 0xb8606a },
  vena: { kind: 'heart', color: 0x5f7fb8 },
};

/* Selectable regions. Key = group, or "group|part-id" for one part of a group (checked first). */
export const REGIONS = {
  vent: {
    label: 'Ventricles',
    vessels: ['LAD', 'LCX', 'RCA'],
    text: 'The LAD supplies the front wall and most of the septum, the LCX the left side and back wall, ' +
      'and the RCA the right ventricle and the underside (usual right-dominant pattern; this varies between people).',
    features: ['EF-TTE', 'Region RWMA'],
  },
  'atria|FJ2439': {
    label: 'Right atrium',
    vessels: ['RCA'],
    text: 'Usually supplied by the RCA, which in most people also feeds the sinoatrial node, the heart\u2019s natural pacemaker.',
  },
  'atria|FJ2438': {
    label: 'Left atrium',
    vessels: ['LCX'],
    text: 'Mainly supplied by branches of the left circumflex artery.',
  },
  aorta: {
    label: 'Aorta',
    vessels: ['LAD', 'LCX', 'RCA'],
    text: 'Both coronary arteries start from the aortic root, just above the aortic valve: the left main trunk (which splits into LAD and LCX) and the RCA.',
  },
  pulm: { label: 'Pulmonary trunk and arteries', vessels: [], text: 'Carry blood from the right ventricle to the lungs. Not a coronary territory and not predicted here.' },
  pvein: { label: 'Pulmonary veins', vessels: [], text: 'Bring oxygenated blood from the lungs to the left atrium. Not a coronary territory and not predicted here.' },
  vena: { label: 'Superior vena cava', vessels: [], text: 'Returns blood from the upper body to the right atrium. Not a coronary territory and not predicted here.' },
};

export const prettyName = (raw) => (raw || '').split('|').pop().replace(/_/g, ' ');
