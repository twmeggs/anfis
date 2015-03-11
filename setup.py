from setuptools import setup, find_packages

setup(name='anfis',
      version='0.1',
      description='Python Adaptive Neuro Fuzzy Inference System',
      url='https://github.com/twmeggs/anfis',
      author='Tim Meggs',
      author_email='twmeggs@gmail.com',
      license='MIT',
      include_package_data = True,
      dependency_links = ["https://github.com/scikit-fuzzy/scikit-fuzzy"],
      keywords = 'anfis, fuzzy logic, neural networks'
      packages=find_packages(),

      install_requires = ['numpy','itertools','copy','scikit-fuzzy']

      classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 3 - Alpha',

        # Indicate who your project is intended for
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',

        # Pick your license as you wish (should match "license" above)
        'License :: OSI Approved :: MIT License',

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.6',
        'Programming Language :: Python :: 2.7',]
)
