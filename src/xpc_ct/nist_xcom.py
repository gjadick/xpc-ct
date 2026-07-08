"""
NIST XCOM material definitions.

Source: 
    Hubbell, J.H. and Seltzer, S.M. (2004), Tables of X-Ray Mass Attenuation Coefficients 
    and Mass Energy-Absorption Coefficients (version 1.4). [Online] Available: 
    http://physics.nist.gov/xaamdi  [2026, May 27]. 
    National Institute of Standards and Technology, Gaithersburg, MD.

Table URLs:
    https://physics.nist.gov/PhysRefData/XrayMassCoef/tab1.html
    https://physics.nist.gov/PhysRefData/XrayMassCoef/tab2.html
        
Notes:
    - density is in g/cm^3
    - excitation_energy_eV is the mean excitation energy I in eV
    - matcomp gives elemental mass fractions in the parser-compatible format
      used by pbxpci_ct.materials.parse_mass_fraction_string()
"""


NIST_XCOM_ALIASES = {
    'adipose': 'adipose_tissue_icru44',
    'air': 'air_dry_near_sea_level',
    'aluminum': 'Al',
    'bone': 'bone_cortical_icru44',
    'breast': 'breast_tissue_icru44',
    'pmma': 'polymethyl_methacrylate',
    'tissue': 'tissue_soft_icru44',
    'water': 'water_liquid',
}


NIST_XCOM_MATERIALS = {
    'a150_tissue_equivalent_plastic': {
        'name': 'A-150 Tissue-Equivalent Plastic',
        'z_over_a': 0.54903,
        'excitation_energy_eV': 65.1,
        'density': 1.127,
        'matcomp': 'H(0.101330)C(0.775498)N(0.035057)O(0.052315)F(0.017423)Ca(0.018377)',
    },
    'adipose_tissue_icru44': {
        'name': 'Adipose Tissue (ICRU-44)',
        'z_over_a': 0.55579,
        'excitation_energy_eV': 64.8,
        'density': 0.950,
        'matcomp': 'H(0.114000)C(0.598000)N(0.007000)O(0.278000)Na(0.001000)S(0.001000)Cl(0.001000)',
    },
    'air_dry_near_sea_level': {
        'name': 'Air, Dry (near sea level)',
        'z_over_a': 0.49919,
        'excitation_energy_eV': 85.7,
        'density': 1.205e-3,
        'matcomp': 'C(0.000124)N(0.755268)O(0.231781)Ar(0.012827)',
    },
    'alanine': {
        'name': 'Alanine',
        'z_over_a': 0.53876,
        'excitation_energy_eV': 71.9,
        'density': 1.424,
        'matcomp': 'H(0.079192)C(0.404437)N(0.157213)O(0.359157)',
    },
    'b100_bone_equivalent_plastic': {
        'name': 'B-100 Bone-Equivalent Plastic',
        'z_over_a': 0.52740,
        'excitation_energy_eV': 85.9,
        'density': 1.450,
        'matcomp': 'H(0.065473)C(0.536942)N(0.021500)O(0.032084)F(0.167415)Ca(0.176585)',
    },
    'bakelite': {
        'name': 'Bakelite',
        'z_over_a': 0.52792,
        'excitation_energy_eV': 72.4,
        'density': 1.250,
        'matcomp': 'H(0.057444)C(0.774589)O(0.167968)',
    },
    'blood_whole_icru44': {
        'name': 'Blood, Whole (ICRU-44)',
        'z_over_a': 0.54999,
        'excitation_energy_eV': 75.2,
        'density': 1.060,
        'matcomp': 'H(0.102000)C(0.110000)N(0.033000)O(0.745000)Na(0.001000)P(0.001000)S(0.002000)Cl(0.003000)K(0.002000)Fe(0.001000)',
    },
    'bone_cortical_icru44': {
        'name': 'Bone, Cortical (ICRU-44)',
        'z_over_a': 0.51478,
        'excitation_energy_eV': 112.0,
        'density': 1.920,
        'matcomp': 'H(0.034000)C(0.155000)N(0.042000)O(0.435000)Na(0.001000)Mg(0.002000)P(0.103000)S(0.003000)Ca(0.225000)',
    },
    'brain_grey_white_matter_icru44': {
        'name': 'Brain, Grey/White Matter (ICRU-44)',
        'z_over_a': 0.55239,
        'excitation_energy_eV': 73.9,
        'density': 1.040,
        'matcomp': 'H(0.107000)C(0.145000)N(0.022000)O(0.712000)Na(0.002000)P(0.004000)S(0.002000)Cl(0.003000)K(0.003000)',
    },
    'breast_tissue_icru44': {
        'name': 'Breast Tissue (ICRU-44)',
        'z_over_a': 0.55196,
        'excitation_energy_eV': 70.3,
        'density': 1.020,
        'matcomp': 'H(0.106000)C(0.332000)N(0.030000)O(0.527000)Na(0.001000)P(0.001000)S(0.002000)Cl(0.001000)',
    },
    'c552_air_equivalent_plastic': {
        'name': 'C-552 Air-equivalent Plastic',
        'z_over_a': 0.49969,
        'excitation_energy_eV': 86.8,
        'density': 1.760,
        'matcomp': 'H(0.024681)C(0.501610)O(0.004527)F(0.465209)Si(0.003973)',
    },
    'cadmium_telluride': {
        'name': 'Cadmium Telluride',
        'z_over_a': 0.41665,
        'excitation_energy_eV': 539.3,
        'density': 6.200,
        'matcomp': 'Cd(0.468358)Te(0.531642)',
    },
    'calcium_fluoride': {
        'name': 'Calcium Fluoride',
        'z_over_a': 0.48671,
        'excitation_energy_eV': 166.0,
        'density': 3.180,
        'matcomp': 'F(0.486672)Ca(0.513328)',
    },
    'calcium_sulfate': {
        'name': 'Calcium Sulfate',
        'z_over_a': 0.49948,
        'excitation_energy_eV': 152.3,
        'density': 2.960,
        'matcomp': 'O(0.470081)S(0.235534)Ca(0.294385)',
    },
    'ceric_ammonium_sulfate_solution': {
        'name': '15 mmol L-1 Ceric Ammonium Sulfate Solution',
        'z_over_a': 0.55282,
        'excitation_energy_eV': 76.7,
        'density': 1.030,
        'matcomp': 'H(0.107694)N(0.000816)O(0.875172)S(0.014279)Ce(0.002040)',
    },
    'cesium_iodide': {
        'name': 'Cesium Iodide',
        'z_over_a': 0.41569,
        'excitation_energy_eV': 553.1,
        'density': 4.510,
        'matcomp': 'I(0.488451)Cs(0.511549)',
    },
    'concrete_ordinary': {
        'name': 'Concrete, Ordinary',
        'z_over_a': 0.50932,
        'excitation_energy_eV': 124.5,
        'density': 2.300,
        'matcomp': 'H(0.022100)C(0.002484)O(0.574930)Na(0.015208)Mg(0.001266)Al(0.019953)Si(0.304627)K(0.010045)Ca(0.042951)Fe(0.006435)',
    },
    'concrete_barite': {
        'name': 'Concrete, Barite (TYPE BA)',
        'z_over_a': 0.45714,
        'excitation_energy_eV': 248.2,
        'density': 3.350,
        'matcomp': 'H(0.003585)O(0.311622)Mg(0.001195)Al(0.004183)Si(0.010457)S(0.107858)Ca(0.050194)Fe(0.047505)Ba(0.463400)',
    },
    'eye_lens_icru44': {
        'name': 'Eye Lens (ICRU-44)',
        'z_over_a': 0.54709,
        'excitation_energy_eV': 74.3,
        'density': 1.070,
        'matcomp': 'H(0.096000)C(0.195000)N(0.057000)O(0.646000)Na(0.001000)P(0.001000)S(0.003000)Cl(0.001000)',
    },
    'ferrous_sulfate_standard_fricke': {
        'name': 'Ferrous Sulfate Standard Fricke',
        'z_over_a': 0.55334,
        'excitation_energy_eV': 76.3,
        'density': 1.024,
        'matcomp': 'H(0.108376)O(0.878959)Na(0.000022)S(0.012553)Cl(0.000035)Fe(0.000055)',
    },
    'gadolinium_oxysulfide': {
        'name': 'Gadolinium Oxysulfide',
        'z_over_a': 0.42265,
        'excitation_energy_eV': 493.3,
        'density': 7.440,
        'matcomp': 'O(0.084527)S(0.084704)Gd(0.830769)',
    },
    'gafchromic_sensor': {
        'name': 'Gafchromic Sensor',
        'z_over_a': 0.54384,
        'excitation_energy_eV': 67.2,
        'density': 1.300,
        'matcomp': 'H(0.089700)C(0.605800)N(0.112200)O(0.192300)',
    },
    'gallium_arsenide': {
        'name': 'Gallium Arsenide',
        'z_over_a': 0.44246,
        'excitation_energy_eV': 384.9,
        'density': 5.310,
        'matcomp': 'Ga(0.482030)As(0.517970)',
    },
    'glass_borosilicate_pyrex': {
        'name': 'Glass, Borosilicate (Pyrex)',
        'z_over_a': 0.49707,
        'excitation_energy_eV': 134.0,
        'density': 2.230,
        'matcomp': 'B(0.040066)O(0.539559)Na(0.028191)Al(0.011644)Si(0.377220)K(0.003321)',
    },
    'glass_lead': {
        'name': 'Glass, Lead',
        'z_over_a': 0.42101,
        'excitation_energy_eV': 526.4,
        'density': 6.220,
        'matcomp': 'O(0.156453)Si(0.080866)Ti(0.008092)As(0.002651)Pb(0.751938)',
    },
    'lithium_fluoride': {
        'name': 'Lithium Fluoride',
        'nist_name': 'Lithium Fluride',
        'z_over_a': 0.46262,
        'excitation_energy_eV': 94.0,
        'density': 2.635,
        'matcomp': 'Li(0.267585)F(0.732415)',
    },
    'lithium_tetraborate': {
        'name': 'Lithium Tetraborate',
        'z_over_a': 0.48485,
        'excitation_energy_eV': 94.6,
        'density': 2.440,
        'matcomp': 'Li(0.082081)B(0.255715)O(0.662204)',
    },
    'lung_tissue_icru44': {
        'name': 'Lung Tissue (ICRU-44)',
        'z_over_a': 0.55048,
        'excitation_energy_eV': 75.2,
        'density': 1.050,
        'matcomp': 'H(0.103000)C(0.105000)N(0.031000)O(0.749000)Na(0.002000)P(0.002000)S(0.003000)Cl(0.003000)K(0.002000)',
    },
    'magnesium_tetraborate': {
        'name': 'Magnesium Tetraborate',
        'nist_name': 'Magnesium Tetroborate',
        'z_over_a': 0.49012,
        'excitation_energy_eV': 108.3,
        'density': 2.530,
        'matcomp': 'B(0.240870)O(0.623762)Mg(0.135367)',
    },
    'mercuric_iodide': {
        'name': 'Mercuric Iodide',
        'z_over_a': 0.40933,
        'excitation_energy_eV': 684.5,
        'density': 6.360,
        'matcomp': 'I(0.558560)Hg(0.441440)',
    },
    'muscle_skeletal_icru44': {
        'name': 'Muscle, Skeletal (ICRU-44)',
        'z_over_a': 0.55000,
        'excitation_energy_eV': 74.6,
        'density': 1.050,
        'matcomp': 'H(0.102000)C(0.143000)N(0.034000)O(0.710000)Na(0.001000)P(0.002000)S(0.003000)Cl(0.001000)K(0.004000)',
    },
    'ovary_icru44': {
        'name': 'Ovary (ICRU-44)',
        'z_over_a': 0.55149,
        'excitation_energy_eV': 75.0,
        'density': 1.050,
        'matcomp': 'H(0.105000)C(0.093000)N(0.024000)O(0.768000)Na(0.002000)P(0.002000)S(0.002000)Cl(0.002000)K(0.002000)',
    },
    'photographic_emulsion_kodak_type_aa': {
        'name': 'Photographic Emulsion (Kodak Type AA)',
        'z_over_a': 0.48176,
        'excitation_energy_eV': 179.0,
        'density': 2.200,
        'matcomp': 'H(0.030500)C(0.210700)N(0.072100)O(0.163200)Br(0.222800)Ag(0.300700)',
    },
    'photographic_emulsion_standard_nuclear': {
        'name': 'Photographic Emulsion (Standard Nuclear)',
        'z_over_a': 0.45453,
        'excitation_energy_eV': 331.0,
        'density': 3.815,
        'matcomp': 'H(0.014100)C(0.072261)N(0.019320)O(0.066101)S(0.001890)Br(0.349104)Ag(0.474105)I(0.003120)',
    },
    'plastic_scintillator_vinyltoluene': {
        'name': 'Plastic Scintillator, Vinyltoluene',
        'z_over_a': 0.54141,
        'excitation_energy_eV': 64.7,
        'density': 1.032,
        'matcomp': 'H(0.085000)C(0.915000)',
    },
    'polyethylene': {
        'name': 'Polyethylene',
        'z_over_a': 0.57033,
        'excitation_energy_eV': 57.4,
        'density': 0.930,
        'matcomp': 'H(0.143716)C(0.856284)',
    },
    'polyethylene_terephthalate_mylar': {
        'name': 'Polyethylene Terephthalate, (Mylar)',
        'z_over_a': 0.52037,
        'excitation_energy_eV': 78.7,
        'density': 1.380,
        'matcomp': 'H(0.041960)C(0.625016)O(0.333024)',
    },
    'polymethyl_methacrylate': {
        'name': 'Polymethyl Methacrylate',
        'z_over_a': 0.53937,
        'excitation_energy_eV': 74.0,
        'density': 1.190,
        'matcomp': 'H(0.080541)C(0.599846)O(0.319613)',
    },
    'polystyrene': {
        'name': 'Polystyrene',
        'z_over_a': 0.53768,
        'excitation_energy_eV': 68.7,
        'density': 1.060,
        'matcomp': 'H(0.077421)C(0.922579)',
    },
    'polytetrafluoroethylene_teflon': {
        'name': 'Polytetrafluoroethylene, (Teflon)',
        'z_over_a': 0.47993,
        'excitation_energy_eV': 99.1,
        'density': 2.250,
        'matcomp': 'C(0.240183)F(0.759818)',
    },
    'polyvinyl_chloride': {
        'name': 'Polyvinyl Chloride',
        'z_over_a': 0.51201,
        'excitation_energy_eV': 108.2,
        'density': 1.406,
        'matcomp': 'H(0.048382)C(0.384361)Cl(0.567257)',
    },
    'radiochromic_dye_film_nylon_base': {
        'name': 'Radiochromic Dye Film, Nylon Base',
        'z_over_a': 0.54987,
        'excitation_energy_eV': 64.5,
        'density': 1.080,
        'matcomp': 'H(0.101996)C(0.654396)N(0.098915)O(0.144693)',
    },
    'testis_icru44': {
        'name': 'Testis (ICRU-44)',
        'z_over_a': 0.55200,
        'excitation_energy_eV': 74.7,
        'density': 1.040,
        'matcomp': 'H(0.106000)C(0.099000)N(0.020000)O(0.766000)Na(0.002000)P(0.001000)S(0.002000)Cl(0.002000)K(0.002000)',
    },
    'tissue_soft_icru44': {
        'name': 'Tissue, Soft (ICRU-44)',
        'z_over_a': 0.54996,
        'excitation_energy_eV': 74.7,
        'density': 1.060,
        'matcomp': 'H(0.102000)C(0.143000)N(0.034000)O(0.708000)Na(0.002000)P(0.003000)S(0.003000)Cl(0.002000)K(0.003000)',
    },
    'tissue_soft_icru_four_component': {
        'name': 'Tissue, Soft (ICRU Four-Component)',
        'z_over_a': 0.54975,
        'excitation_energy_eV': 74.9,
        'density': 1.000,
        'matcomp': 'H(0.101174)C(0.111000)N(0.026000)O(0.761826)',
    },
    'tissue_equivalent_gas_methane_based': {
        'name': 'Tissue-Equivalent Gas, Methane Based',
        'z_over_a': 0.54992,
        'excitation_energy_eV': 61.2,
        'density': 1.064e-3,
        'matcomp': 'H(0.101873)C(0.456177)N(0.035172)O(0.406778)',
    },
    'tissue_equivalent_gas_propane_based': {
        'name': 'Tissue-Equivalent Gas, Propane Based',
        'z_over_a': 0.55027,
        'excitation_energy_eV': 59.5,
        'density': 1.826e-3,
        'matcomp': 'H(0.102676)C(0.568937)N(0.035022)O(0.293365)',
    },
    'water_liquid': {
        'name': 'Water, Liquid',
        'z_over_a': 0.55508,
        'excitation_energy_eV': 75.0,
        'density': 1.000,
        'matcomp': 'H(0.111898)O(0.888102)',
    },
}


NIST_XCOM_ELEMENTS = {
    'H': {
        'Z': 1,
        'symbol': 'H',
        'name': 'Hydrogen',
        'z_over_a': 0.99212,
        'excitation_energy_eV': 19.2,
        'density': 8.375e-5,
    },
    'He': {
        'Z': 2,
        'symbol': 'He',
        'name': 'Helium',
        'z_over_a': 0.49968,
        'excitation_energy_eV': 41.8,
        'density': 1.663e-4,
    },
    'Li': {
        'Z': 3,
        'symbol': 'Li',
        'name': 'Lithium',
        'z_over_a': 0.43221,
        'excitation_energy_eV': 40.0,
        'density': 5.340e-1,
    },
    'Be': {
        'Z': 4,
        'symbol': 'Be',
        'name': 'Beryllium',
        'z_over_a': 0.44384,
        'excitation_energy_eV': 63.7,
        'density': 1.848,
    },
    'B': {
        'Z': 5,
        'symbol': 'B',
        'name': 'Boron',
        'z_over_a': 0.46245,
        'excitation_energy_eV': 76.0,
        'density': 2.370,
    },
    'C': {
        'Z': 6,
        'symbol': 'C',
        'name': 'Carbon, Graphite',
        'z_over_a': 0.49954,
        'excitation_energy_eV': 78.0,
        'density': 1.700,
    },
    'N': {
        'Z': 7,
        'symbol': 'N',
        'name': 'Nitrogen',
        'z_over_a': 0.49976,
        'excitation_energy_eV': 82.0,
        'density': 1.165e-3,
    },
    'O': {
        'Z': 8,
        'symbol': 'O',
        'name': 'Oxygen',
        'z_over_a': 0.50002,
        'excitation_energy_eV': 95.0,
        'density': 1.332e-3,
    },
    'F': {
        'Z': 9,
        'symbol': 'F',
        'name': 'Fluorine',
        'z_over_a': 0.47372,
        'excitation_energy_eV': 115.0,
        'density': 1.580e-3,
    },
    'Ne': {
        'Z': 10,
        'symbol': 'Ne',
        'name': 'Neon',
        'z_over_a': 0.49555,
        'excitation_energy_eV': 137.0,
        'density': 8.385e-4,
    },
    'Na': {
        'Z': 11,
        'symbol': 'Na',
        'name': 'Sodium',
        'z_over_a': 0.47847,
        'excitation_energy_eV': 149.0,
        'density': 9.710e-1,
    },
    'Mg': {
        'Z': 12,
        'symbol': 'Mg',
        'name': 'Magnesium',
        'z_over_a': 0.49373,
        'excitation_energy_eV': 156.0,
        'density': 1.740,
    },
    'Al': {
        'Z': 13,
        'symbol': 'Al',
        'name': 'Aluminum',
        'z_over_a': 0.48181,
        'excitation_energy_eV': 166.0,
        'density': 2.699,
    },
    'Si': {
        'Z': 14,
        'symbol': 'Si',
        'name': 'Silicon',
        'z_over_a': 0.49848,
        'excitation_energy_eV': 173.0,
        'density': 2.330,
    },
    'P': {
        'Z': 15,
        'symbol': 'P',
        'name': 'Phosphorus',
        'z_over_a': 0.48428,
        'excitation_energy_eV': 173.0,
        'density': 2.200,
    },
    'S': {
        'Z': 16,
        'symbol': 'S',
        'name': 'Sulfur',
        'z_over_a': 0.49897,
        'excitation_energy_eV': 180.0,
        'density': 2.000,
    },
    'Cl': {
        'Z': 17,
        'symbol': 'Cl',
        'name': 'Chlorine',
        'z_over_a': 0.47951,
        'excitation_energy_eV': 174.0,
        'density': 2.995e-3,
    },
    'Ar': {
        'Z': 18,
        'symbol': 'Ar',
        'name': 'Argon',
        'z_over_a': 0.45059,
        'excitation_energy_eV': 188.0,
        'density': 1.662e-3,
    },
    'K': {
        'Z': 19,
        'symbol': 'K',
        'name': 'Potassium',
        'z_over_a': 0.48595,
        'excitation_energy_eV': 190.0,
        'density': 8.620e-1,
    },
    'Ca': {
        'Z': 20,
        'symbol': 'Ca',
        'name': 'Calcium',
        'z_over_a': 0.49903,
        'excitation_energy_eV': 191.0,
        'density': 1.550,
    },
    'Sc': {
        'Z': 21,
        'symbol': 'Sc',
        'name': 'Scandium',
        'z_over_a': 0.46712,
        'excitation_energy_eV': 216.0,
        'density': 2.989,
    },
    'Ti': {
        'Z': 22,
        'symbol': 'Ti',
        'name': 'Titanium',
        'z_over_a': 0.45948,
        'excitation_energy_eV': 233.0,
        'density': 4.540,
    },
    'V': {
        'Z': 23,
        'symbol': 'V',
        'name': 'Vanadium',
        'z_over_a': 0.45150,
        'excitation_energy_eV': 245.0,
        'density': 6.110,
    },
    'Cr': {
        'Z': 24,
        'symbol': 'Cr',
        'name': 'Chromium',
        'z_over_a': 0.46157,
        'excitation_energy_eV': 257.0,
        'density': 7.180,
    },
    'Mn': {
        'Z': 25,
        'symbol': 'Mn',
        'name': 'Manganese',
        'z_over_a': 0.45506,
        'excitation_energy_eV': 272.0,
        'density': 7.440,
    },
    'Fe': {
        'Z': 26,
        'symbol': 'Fe',
        'name': 'Iron',
        'z_over_a': 0.46556,
        'excitation_energy_eV': 286.0,
        'density': 7.874,
    },
    'Co': {
        'Z': 27,
        'symbol': 'Co',
        'name': 'Cobalt',
        'z_over_a': 0.45815,
        'excitation_energy_eV': 297.0,
        'density': 8.900,
    },
    'Ni': {
        'Z': 28,
        'symbol': 'Ni',
        'name': 'Nickel',
        'z_over_a': 0.47708,
        'excitation_energy_eV': 311.0,
        'density': 8.902,
    },
    'Cu': {
        'Z': 29,
        'symbol': 'Cu',
        'name': 'Copper',
        'z_over_a': 0.45636,
        'excitation_energy_eV': 322.0,
        'density': 8.960,
    },
    'Zn': {
        'Z': 30,
        'symbol': 'Zn',
        'name': 'Zinc',
        'z_over_a': 0.45879,
        'excitation_energy_eV': 330.0,
        'density': 7.133,
    },
    'Ga': {
        'Z': 31,
        'symbol': 'Ga',
        'name': 'Gallium',
        'z_over_a': 0.44462,
        'excitation_energy_eV': 334.0,
        'density': 5.904,
    },
    'Ge': {
        'Z': 32,
        'symbol': 'Ge',
        'name': 'Germanium',
        'z_over_a': 0.44071,
        'excitation_energy_eV': 350.0,
        'density': 5.323,
    },
    'As': {
        'Z': 33,
        'symbol': 'As',
        'name': 'Arsenic',
        'z_over_a': 0.44046,
        'excitation_energy_eV': 347.0,
        'density': 5.730,
    },
    'Se': {
        'Z': 34,
        'symbol': 'Se',
        'name': 'Selenium',
        'z_over_a': 0.43060,
        'excitation_energy_eV': 348.0,
        'density': 4.500,
    },
    'Br': {
        'Z': 35,
        'symbol': 'Br',
        'name': 'Bromine',
        'z_over_a': 0.43803,
        'excitation_energy_eV': 343.0,
        'density': 7.072e-3,
    },
    'Kr': {
        'Z': 36,
        'symbol': 'Kr',
        'name': 'Krypton',
        'z_over_a': 0.42959,
        'excitation_energy_eV': 352.0,
        'density': 3.478e-3,
    },
    'Rb': {
        'Z': 37,
        'symbol': 'Rb',
        'name': 'Rubidium',
        'z_over_a': 0.43291,
        'excitation_energy_eV': 363.0,
        'density': 1.532,
    },
    'Sr': {
        'Z': 38,
        'symbol': 'Sr',
        'name': 'Strontium',
        'z_over_a': 0.43369,
        'excitation_energy_eV': 366.0,
        'density': 2.540,
    },
    'Y': {
        'Z': 39,
        'symbol': 'Y',
        'name': 'Yttrium',
        'z_over_a': 0.43867,
        'excitation_energy_eV': 379.0,
        'density': 4.469,
    },
    'Zr': {
        'Z': 40,
        'symbol': 'Zr',
        'name': 'Zirconium',
        'z_over_a': 0.43848,
        'excitation_energy_eV': 393.0,
        'density': 6.506,
    },
    'Nb': {
        'Z': 41,
        'symbol': 'Nb',
        'name': 'Niobium',
        'z_over_a': 0.44130,
        'excitation_energy_eV': 417.0,
        'density': 8.570,
    },
    'Mo': {
        'Z': 42,
        'symbol': 'Mo',
        'name': 'Molybdenum',
        'z_over_a': 0.43777,
        'excitation_energy_eV': 424.0,
        'density': 10.22,
    },
    'Tc': {
        'Z': 43,
        'symbol': 'Tc',
        'name': 'Technetium',
        'z_over_a': 0.43919,
        'excitation_energy_eV': 428.0,
        'density': 11.50,
    },
    'Ru': {
        'Z': 44,
        'symbol': 'Ru',
        'name': 'Ruthenium',
        'z_over_a': 0.43534,
        'excitation_energy_eV': 441.0,
        'density': 12.41,
    },
    'Rh': {
        'Z': 45,
        'symbol': 'Rh',
        'name': 'Rhodium',
        'z_over_a': 0.43729,
        'excitation_energy_eV': 449.0,
        'density': 12.41,
    },
    'Pd': {
        'Z': 46,
        'symbol': 'Pd',
        'name': 'Palladium',
        'z_over_a': 0.43225,
        'excitation_energy_eV': 470.0,
        'density': 12.02,
    },
    'Ag': {
        'Z': 47,
        'symbol': 'Ag',
        'name': 'Silver',
        'z_over_a': 0.43572,
        'excitation_energy_eV': 470.0,
        'density': 10.50,
    },
    'Cd': {
        'Z': 48,
        'symbol': 'Cd',
        'name': 'Cadmium',
        'z_over_a': 0.42700,
        'excitation_energy_eV': 469.0,
        'density': 8.650,
    },
    'In': {
        'Z': 49,
        'symbol': 'In',
        'name': 'Indium',
        'z_over_a': 0.42676,
        'excitation_energy_eV': 488.0,
        'density': 7.310,
    },
    'Sn': {
        'Z': 50,
        'symbol': 'Sn',
        'name': 'Tin',
        'z_over_a': 0.42120,
        'excitation_energy_eV': 488.0,
        'density': 7.310,
    },
    'Sb': {
        'Z': 51,
        'symbol': 'Sb',
        'name': 'Antimony',
        'z_over_a': 0.41889,
        'excitation_energy_eV': 487.0,
        'density': 6.691,
    },
    'Te': {
        'Z': 52,
        'symbol': 'Te',
        'name': 'Tellurium',
        'z_over_a': 0.40752,
        'excitation_energy_eV': 485.0,
        'density': 6.240,
    },
    'I': {
        'Z': 53,
        'symbol': 'I',
        'name': 'Iodine',
        'z_over_a': 0.41764,
        'excitation_energy_eV': 491.0,
        'density': 4.930,
    },
    'Xe': {
        'Z': 54,
        'symbol': 'Xe',
        'name': 'Xenon',
        'z_over_a': 0.41130,
        'excitation_energy_eV': 482.0,
        'density': 5.485e-3,
    },
    'Cs': {
        'Z': 55,
        'symbol': 'Cs',
        'name': 'Cesium',
        'z_over_a': 0.41383,
        'excitation_energy_eV': 488.0,
        'density': 1.873,
    },
    'Ba': {
        'Z': 56,
        'symbol': 'Ba',
        'name': 'Barium',
        'z_over_a': 0.40779,
        'excitation_energy_eV': 491.0,
        'density': 3.500,
    },
    'La': {
        'Z': 57,
        'symbol': 'La',
        'name': 'Lanthanum',
        'z_over_a': 0.41035,
        'excitation_energy_eV': 501.0,
        'density': 6.154,
    },
    'Ce': {
        'Z': 58,
        'symbol': 'Ce',
        'name': 'Cerium',
        'z_over_a': 0.41395,
        'excitation_energy_eV': 523.0,
        'density': 6.657,
    },
    'Pr': {
        'Z': 59,
        'symbol': 'Pr',
        'name': 'Praseodymium',
        'z_over_a': 0.41871,
        'excitation_energy_eV': 535.0,
        'density': 6.710,
    },
    'Nd': {
        'Z': 60,
        'symbol': 'Nd',
        'name': 'Neodymium',
        'z_over_a': 0.41597,
        'excitation_energy_eV': 546.0,
        'density': 6.900,
    },
    'Pm': {
        'Z': 61,
        'symbol': 'Pm',
        'name': 'Promethium',
        'z_over_a': 0.42094,
        'excitation_energy_eV': 560.0,
        'density': 7.220,
    },
    'Sm': {
        'Z': 62,
        'symbol': 'Sm',
        'name': 'Samarium',
        'z_over_a': 0.41234,
        'excitation_energy_eV': 574.0,
        'density': 7.460,
    },
    'Eu': {
        'Z': 63,
        'symbol': 'Eu',
        'name': 'Europium',
        'z_over_a': 0.41457,
        'excitation_energy_eV': 580.0,
        'density': 5.243,
    },
    'Gd': {
        'Z': 64,
        'symbol': 'Gd',
        'name': 'Gadolinium',
        'z_over_a': 0.40699,
        'excitation_energy_eV': 591.0,
        'density': 7.900,
    },
    'Tb': {
        'Z': 65,
        'symbol': 'Tb',
        'name': 'Terbium',
        'z_over_a': 0.40900,
        'excitation_energy_eV': 614.0,
        'density': 8.229,
    },
    'Dy': {
        'Z': 66,
        'symbol': 'Dy',
        'name': 'Dysprosium',
        'z_over_a': 0.40615,
        'excitation_energy_eV': 628.0,
        'density': 8.550,
    },
    'Ho': {
        'Z': 67,
        'symbol': 'Ho',
        'name': 'Holmium',
        'z_over_a': 0.40623,
        'excitation_energy_eV': 650.0,
        'density': 8.795,
    },
    'Er': {
        'Z': 68,
        'symbol': 'Er',
        'name': 'Erbium',
        'z_over_a': 0.40655,
        'excitation_energy_eV': 658.0,
        'density': 9.066,
    },
    'Tm': {
        'Z': 69,
        'symbol': 'Tm',
        'name': 'Thulium',
        'z_over_a': 0.40844,
        'excitation_energy_eV': 674.0,
        'density': 9.321,
    },
    'Yb': {
        'Z': 70,
        'symbol': 'Yb',
        'name': 'Ytterbium',
        'z_over_a': 0.40453,
        'excitation_energy_eV': 684.0,
        'density': 6.730,
    },
    'Lu': {
        'Z': 71,
        'symbol': 'Lu',
        'name': 'Lutetium',
        'z_over_a': 0.40579,
        'excitation_energy_eV': 694.0,
        'density': 9.840,
    },
    'Hf': {
        'Z': 72,
        'symbol': 'Hf',
        'name': 'Hafnium',
        'z_over_a': 0.40338,
        'excitation_energy_eV': 705.0,
        'density': 13.31,
    },
    'Ta': {
        'Z': 73,
        'symbol': 'Ta',
        'name': 'Tantalum',
        'z_over_a': 0.40343,
        'excitation_energy_eV': 718.0,
        'density': 16.65,
    },
    'W': {
        'Z': 74,
        'symbol': 'W',
        'name': 'Tungsten',
        'z_over_a': 0.40250,
        'excitation_energy_eV': 727.0,
        'density': 19.30,
    },
    'Re': {
        'Z': 75,
        'symbol': 'Re',
        'name': 'Rhenium',
        'z_over_a': 0.40278,
        'excitation_energy_eV': 736.0,
        'density': 21.02,
    },
    'Os': {
        'Z': 76,
        'symbol': 'Os',
        'name': 'Osmium',
        'z_over_a': 0.39958,
        'excitation_energy_eV': 746.0,
        'density': 22.57,
    },
    'Ir': {
        'Z': 77,
        'symbol': 'Ir',
        'name': 'Iridium',
        'z_over_a': 0.40058,
        'excitation_energy_eV': 757.0,
        'density': 22.42,
    },
    'Pt': {
        'Z': 78,
        'symbol': 'Pt',
        'name': 'Platinum',
        'z_over_a': 0.39984,
        'excitation_energy_eV': 790.0,
        'density': 21.45,
    },
    'Au': {
        'Z': 79,
        'symbol': 'Au',
        'name': 'Gold',
        'z_over_a': 0.40108,
        'excitation_energy_eV': 790.0,
        'density': 19.32,
    },
    'Hg': {
        'Z': 80,
        'symbol': 'Hg',
        'name': 'Mercury',
        'z_over_a': 0.39882,
        'excitation_energy_eV': 800.0,
        'density': 13.55,
    },
    'Tl': {
        'Z': 81,
        'symbol': 'Tl',
        'name': 'Thallium',
        'z_over_a': 0.39631,
        'excitation_energy_eV': 810.0,
        'density': 11.72,
    },
    'Pb': {
        'Z': 82,
        'symbol': 'Pb',
        'name': 'Lead',
        'z_over_a': 0.39575,
        'excitation_energy_eV': 823.0,
        'density': 11.35,
    },
    'Bi': {
        'Z': 83,
        'symbol': 'Bi',
        'name': 'Bismuth',
        'z_over_a': 0.39717,
        'excitation_energy_eV': 823.0,
        'density': 9.747,
    },
    'Po': {
        'Z': 84,
        'symbol': 'Po',
        'name': 'Polonium',
        'z_over_a': 0.40195,
        'excitation_energy_eV': 830.0,
        'density': 9.320,
    },
    'At': {
        'Z': 85,
        'symbol': 'At',
        'name': 'Astatine',
        'z_over_a': 0.40479,
        'excitation_energy_eV': 825.0,
        'density': 10.00,
    },
    'Rn': {
        'Z': 86,
        'symbol': 'Rn',
        'name': 'Radon',
        'z_over_a': 0.38736,
        'excitation_energy_eV': 794.0,
        'density': 9.066e-3,
    },
    'Fr': {
        'Z': 87,
        'symbol': 'Fr',
        'name': 'Francium',
        'z_over_a': 0.39010,
        'excitation_energy_eV': 827.0,
        'density': 10.00,
    },
    'Ra': {
        'Z': 88,
        'symbol': 'Ra',
        'name': 'Radium',
        'z_over_a': 0.38934,
        'excitation_energy_eV': 826.0,
        'density': 5.000,
    },
    'Ac': {
        'Z': 89,
        'symbol': 'Ac',
        'name': 'Actinium',
        'z_over_a': 0.39202,
        'excitation_energy_eV': 841.0,
        'density': 10.07,
    },
    'Th': {
        'Z': 90,
        'symbol': 'Th',
        'name': 'Thorium',
        'z_over_a': 0.38787,
        'excitation_energy_eV': 847.0,
        'density': 11.72,
    },
    'Pa': {
        'Z': 91,
        'symbol': 'Pa',
        'name': 'Protactinium',
        'z_over_a': 0.39388,
        'excitation_energy_eV': 878.0,
        'density': 15.37,
    },
    'U': {
        'Z': 92,
        'symbol': 'U',
        'name': 'Uranium',
        'z_over_a': 0.38651,
        'excitation_energy_eV': 890.0,
        'density': 18.95,
    },
}


def _normalize_key(key):
    """Normalize lookup strings for aliases and exact-name matching."""
    return key.strip().lower()


NIST_XCOM_NAME_TO_KEY = {
    _normalize_key(entry['name']): key
    for key, entry in NIST_XCOM_MATERIALS.items()
}

NIST_XCOM_ELEMENT_NAME_TO_SYMBOL = {
    _normalize_key(entry['name']): symbol
    for symbol, entry in NIST_XCOM_ELEMENTS.items()
}


def get_nist_xcom_entry(key):
    """
    Return a NIST XCOM material or element entry.

    Accepted inputs include:
        - material keys, e.g. 'water_liquid'
        - aliases, e.g. 'water', 'pmma', 'aluminum'
        - exact NIST material names, e.g. 'Water, Liquid'
        - element symbols, e.g. 'Al'
        - lowercase element symbols, e.g. 'al'
        - element names, e.g. 'Aluminum'

    Returns
    -------
    entry:
        Shallow copy of the matching entry dictionary.
    """
    key_raw = key.strip()
    key_lower = _normalize_key(key_raw)

    # Aliases can point to either material keys or element symbols.
    key_resolved = NIST_XCOM_ALIASES.get(key_lower, key_raw)
    key_resolved_lower = _normalize_key(key_resolved)

    # Material key lookup.
    if key_resolved_lower in NIST_XCOM_MATERIALS:
        return dict(NIST_XCOM_MATERIALS[key_resolved_lower])

    # Exact NIST material-name lookup.
    if key_resolved_lower in NIST_XCOM_NAME_TO_KEY:
        material_key = NIST_XCOM_NAME_TO_KEY[key_resolved_lower]
        return dict(NIST_XCOM_MATERIALS[material_key])

    # Element symbol lookup, preserving capitalization when possible.
    if key_resolved in NIST_XCOM_ELEMENTS:
        return dict(NIST_XCOM_ELEMENTS[key_resolved])

    # Lowercase element symbol lookup, e.g. 'al' -> 'Al'.
    element_symbol = key_resolved[:1].upper() + key_resolved[1:].lower()

    if element_symbol in NIST_XCOM_ELEMENTS:
        return dict(NIST_XCOM_ELEMENTS[element_symbol])

    # Element-name lookup, e.g. 'aluminum' -> 'Al'.
    if key_resolved_lower in NIST_XCOM_ELEMENT_NAME_TO_SYMBOL:
        element_symbol = NIST_XCOM_ELEMENT_NAME_TO_SYMBOL[key_resolved_lower]
        return dict(NIST_XCOM_ELEMENTS[element_symbol])

    valid_materials = ', '.join(sorted(NIST_XCOM_MATERIALS))
    valid_elements = ', '.join(sorted(NIST_XCOM_ELEMENTS))

    raise KeyError(
        f'Unknown NIST XCOM entry {key!r}.\n'
        f'Valid material keys are: {valid_materials}\n'
        f'Valid element symbols are: {valid_elements}'
    )