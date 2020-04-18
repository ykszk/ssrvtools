from setuptools import setup, find_packages

commands = [
    ('img2mhd', 'image_converter:img2mhd'),
    ('label2mhd', 'image_converter:label2mhd'),
    ('mhd2polygon', 'polygon_converter:main'),
    ('extract_wall', 'cellutils:extract_wall'),
    ('label_cells', 'cellutils:label_cells'),
    ('assign_labels', 'cellutils:assign_labels'),
    ('median_filter', 'image_filter:median_filter'),
]

registratoin_commands = [
    ('coarse_rigid_registration_init', 'coarse_rigid_registration:main'),
    ('fine_rigid_registration_init', 'fine_rigid_registration:main'),
    ('coarse_rigid_registration', 'rigid_registration:coarse_registration'),
    ('fine_rigid_registration', 'rigid_registration:fine_registration'),
    ('apply_rigid_transformations', 'apply_transformation:main'),
    ('apply_rigid_transformation', 'apply_transformation:apply_rigid_transformation'),
    ('apply_inverse_rigid_transformation', 'apply_transformation:apply_inverse_rigid_transformation'),
    ('nonrigid_registration', 'nonrigid_registration:main'),
    ('init_workspace', 'init_workspace:main'),
    ('finalize_registration', 'finalize_registration:main'),
    ('compose_images', 'compose_images:main'),
    ('manual_rigid_registration', 'manual_rigid_registration:main')
]

commands = commands + [(name,'registration.'+func) for name, func in registratoin_commands]

package_name = 'ssrvtools'
setup(
    name=package_name,
    version='0.1.0',
    author='Yuki Suzuki',
    author_email='y-suzuki@radiol.med.osaka-u.ac.jp',
    data_files=[(package_name, ['ssrvtools/colormap.csv'])],
    include_package_data=True,
    packages=find_packages(),
    py_modules=['mhd', 'image_converter', 'polygon_converter',
    'boundingbox', 'cellutils',
    'image_filter'],
    install_requires=[
        'tqdm', 'numpy', 'Pillow', 'scikit-learn',  'vtk',
        'scikit-image', 'scipy', 'matplotlib'
    ],
    entry_points={
        'console_scripts':['{}={}.{}'.format(e[0], package_name, e[1]) for e in commands]
    }
)
