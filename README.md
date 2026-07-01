<!-- Improved compatibility of back to top link: See: https://github.com/othneildrew/Best-README-Template/pull/73 -->
<a id="readme-top"></a>
<!--
*** Thanks for checking out the Best-README-Template. If you have a suggestion
*** that would make this better, please fork the repo and create a pull request
*** or simply open an issue with the tag "enhancement".
*** Don't forget to give the project a star!
*** Thanks again! Now go create something AMAZING! :D
-->



<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->
[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![project_license][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url]



<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/jackbassham/thesis-rough">
    <img src="images/logo.png" alt="Logo" width="80" height="80">
  </a>

<h3 align="center">thesis-rough</h3>

  <p align="center">
    project_description
    <br />
    <a href="https://github.com/jackbassham/thesis-rough"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href="https://github.com/jackbassham/thesis-rough">View Demo</a>
    &middot;
    <a href="https://github.com/jackbassham/thesis-rough/issues/new?labels=bug&template=bug-report---.md">Report Bug</a>
    &middot;
    <a href="https://github.com/jackbassham/thesis-rough/issues/new?labels=enhancement&template=feature-request---.md">Request Feature</a>
  </p>
</div>



<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project

[![Product Name Screen Shot][product-screenshot]](https://example.com)

<!-- Here's a blank template to get started. To avoid retyping too much info, do a search and replace with your text editor for the following: `jackbassham`, `thesis-rough`, `twitter_handle`, `linkedin_username`, `email_client`, `email`, `project_title`, `project_description`, `project_license` -->

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- ### Built With

* [![Next][Next.js]][Next-url]
* [![React][React.js]][React-url]
* [![Vue][Vue.js]][Vue-url]
* [![Angular][Angular.io]][Angular-url]
* [![Svelte][Svelte.dev]][Svelte-url]
* [![Laravel][Laravel.com]][Laravel-url]
* [![Bootstrap][Bootstrap.com]][Bootstrap-url]
* [![JQuery][JQuery.com]][JQuery-url]

<p align="right">(<a href="#readme-top">back to top</a>)</p> -->



<!-- GETTING STARTED -->
## Getting Started

### System Requirements

If running the pipeline on a laptop, it is recommended to use the small Weddell Sea sample dataset configuration, demonstrated [here](#example-data-config-setup). *Model performance will be limitted by this smaller subset because it only includes two years in the training split.*

Downloading the raw datasets, along with storing processed data and model outputs, requires as least **6 GB** of available disk space. The raw data download stage may take several hours, particularly for the ERA5 wind data. 

Running the pipeline with larger datasets and multiple ensemble members can take several days and require substantial memory and disk resources. In these cases, it is recommended to run the pipeline on a remote server using a persistent terminal session (e.g. screen) to prevent interuptions. 

### Prerequisites

* Python 3.10
* Conda
* Git

The pipeline requires Python 3.10 and uses several scientific Python packages, including:

* Cartopy
* cdsapi
* cftime
* cfgrib
* cmocean
* Matplotlib
* NumPy
* PyTorch
* PyYAML
* Requests
* SciPy
* tqdm
* Xarray
* h5netcdf
* h5py

All packages and their compatible versions are managed by a [conda virtual environment](#create-a-conda-virtual-environment) and are specified in `environment.yml`.

### Installation

1. Clone the repo:
   ```sh
   git clone git@github.com:jackbassham/thesis-rough.git
   ```

2. Navigate to the project directory:
   ```sh
   cd thesis-rough
   ```

### (*Optional*) Configure Git Remote URL

If working from a fork and planning to contribute changes, configure your Git remotes to prevent accidentally pushing to the original repository.

1. Change the `origin` remote to your fork:

   ```sh
   git remote set-url origin git@github.com:<YOUR_USERNAME>/thesis-rough.git
   ```

2. Add the original repository as the `upstream` remote to receive future updates:

   ```sh
   git remote add upstream git@github.com:jackbassham/thesis-rough.git
   ```

3. Verify the remote configuration:

   ```sh
   git remote -v
   ```

### Create a Conda Virtual Environment

1. If you haven't already, install Miniconda:
https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html

2. Navigate to the root directory:

    ```sh
    cd thesis-rough
    ```

3. Install the requirements from the `environment.yml` file (included in the repo) in a new environment. The environment is named `thesisrough` by default:

    ```sh
    conda env create -f environment.yml
    ```

4. Activate the environment before running code:

    ```sh
    conda activate thesisrough
    ```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Data Access

### Option A: Download Sample Set *(provided)*
1. Download a subset of raw data covering seven years of ice velocity, wind, and ice concentration
in the Weddell Sea from this [google drive](https://drive.google.com/drive/folders/1JDOTHjX9MqDvm9TemHdnLsqSZsL9ZQoq?usp=share_link).

2. Move the folder into the root of the repo, ensuring the directory structure:
```sh
thesis-rough/data/raw
```

3. Configure data parameters to match the [example](#1-configure-data-parameters), keeping the default data root directory: 
```yaml
user_data_root: null
```

4. [Start the ML pipeline](#option-b-run-partial-pipeline) from the `regrid_ice_vel` step with the raw data version `--timestamp_raw 01010001_0001`:

```
python -m run_pipeline --start regrid_ice_vel --timestamp_raw 01010001_0001
```
*Note* The timestamp versions for data processed in the remaining pipeline steps will reflect the current timestamp
unless specified otherwise in the command line.

### Option B: Download Data Via Pipeline

### For NSIDC Data Access *(Sea Ice Velocity, Sea Ice Concentration)*:
1. [Register here](https://urs.earthdata.nasa.gov/users/new) for a free Nasa Earthdata account. 

2. Create a `.netrc` file in your home directory for Earthdata login information *(Mac/Linux)*:
```sh
vim ~/.netrc
```
Type `i` for insert and enter the following:
```sh
machine urs.earthdata.nasa.gov
login <YOUR_LOGIN>
password <YOUR_PASSWORD>
```
Hit `esc` and enter `:wq` to write the file.

3. Enable permissions:
```sh
chmod 600 ~/.netrc
```

4. Now the repo code will automatically recognnize the credentials and authenticate the requests.


### For ERA5 Data Access *(Wind)*:
1. [Register here](https://accounts.ecmwf.int/auth/realms/ecmwf/login-actions/registration?client_id=cds&tab_id=Tc-gbldHPtE) for a free Climate Data Store (ECMWF) account.

2. Copy the API key from your [profile](https://cds.climate.copernicus.eu/profile)

3. Create a `.cdsapirc` file in your home directory with your API Key *(Mac/Linux)*:
```sh
vim ~/.cdspirc
```
Type `i` for insert and enter the following:
```sh
url: https://cds.climate.copernicus.eu/api
key: <YOUR_API_KEY>
```
Hit `esc` and enter `:wq` to write the file.

4. Accept the dataset agreement [here](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels?tab=download) before runnning code.

*Note: Click on the `CC-BY license` under 'Terms of use', towards the bottom of the page to accept the agreement.*

5. Now the repo code will automatically recoginize the API key for the request.


## Data Structure

Data used for training and evaluation are stored to disk in a root data directory within the repository (unless 'root_directory' is speficied to another location, ie: a scratch directory). Data are currently stored to disk for multiple processing stages. 

Data used for training and evaluation are stored to disk for multiple stages under the following directory hiercy: 

```text
<data_root>/
├── <raw>/
├── <regrid>/
├── <mask_norm>/
├── <model_input>/
├── <model_output>/
    ├── <ps>/
    ├── <lr_cf>/
    ├── <lr_cf_wtd>/
    ├── <cnn>/
    ├── <cnn_wtd>/

<plots>
├── <quick_eval>/
    ├── <ps>/
    ├── <lr_cf>/
    ├── <lr_cf_wtd>/
    ├── <cnn>/
    ├── <cnn_wtd>/
```

**Description:**

*Stages*
- `raw` - Original datasets, downloaded from the source
- `regrid` - Data projected to a common grid within configuration bounds
- `mask-norm` - Masked and normalized data
- `model_input` - Processed training, validation, and testing inputs
- `model_output` - Model weights and predictions

*Models*
- `ps` - Persistence
- `lr_cf` - Closed Form Linear Regression
- `lr_cf_wtd` - Weighted Closed Form Linear Regression
- `cnn` - CNN (via PyTorch)
- `cnn_wtd` - Weighted CNN (via PyTorch)

*Plots*
- `quick_eval` - Spatial plots generated for post-training model evaluation metrics

Versioned datasets then are stored within each stage or model under the following hierchy:

```text
 <data_stage>/
 ├── <hemisphere>/
 │   ├── <timestamp>/
 │   │   └── <data_file>
```

**Description:**
- `hemisphere` - Defines the data subregion. Uses 'south' (Southern Ocean) or 'north' (Arctic Ocean).
- `timestamp` - Version of data assigned at runtime using format 'MMDDYY_HHMM'. Timestamp flexibility and options are explained further in *Usage*.
- `data_file` - All data files are currently stored as numpy arrays (.npz), with the acception of CNN weights (.pth). 

<!-- USAGE EXAMPLES -->
## Usage

<!-- TODO 1. Before getting started, make accounts for to access data..., make a file with account info -->

### 1. Configure Dataset Parameters

Parameters dictating the spatial and temporal coverage of the data used for the ML pipeline,
as well as the path to the root data directory can be modified in the yaml file `./_00_config/user_config.yaml`.
1. Change the active preset:
```yaml
active_preset: <preset>
```
A number of preset configurations exist:
  * `weddell_sea_test` Use this small sample configuration for minimal runtime, memory, and disk requirements; or when using the [sample raw dataset](#option-a-download-sample-set-provided).
  * `full_southern_ocean`, `full_arctic` Use these larger subsets for full reprodcution. *Caution: Requires extensive runtime, memory, and disk resources.*
  * `custom_config` Replace the `null` values to create a custom data subset.

2. *(Optional)* Specify the path to an external data directory by changing `user_data_root` from `null` to the desired path. Leaving this `null` defaults to the project's root directory as the data storage path. 


<!-- **Parameter Descriptions**

- `hemisphere` Defines the region of interest. Use 'south' for Southern Ocean forecasts or 'north' for Arctic Ocean forecasts.

- `year_range` Tuple defining the temporal range of the dataset used for training and evaluation. Maximum supported range is (1989, 2024).

- `latitude_bounds` Tuple defining the meridional bounds (in degrees) of the dataset used. Entered as (southernmost, northernmost). Use negative values for Southern Hemisphere (degrees South). Maximum supported Southern/ Northern Hemisphere ranges are (-90, -40)/ (31, 90). 

- `longitude_bounds` Tuple defining the zonal bounds (in degrees) of the dataset used. Entered as (westernmost, easternmost). Maximum supported range is (-180, 180). Use (-180, 0) for degrees West and (0, 180) for degrees East

- `grid_resolution` Float or int defining the resolution (in kilometers) of the data projection onto a regular latitude longitude grid. Recommended use is 25 km, based on the raw resolution of the sea ice velocity data.  -->


### Option A: Run Full Pipeline

To run the machine learning pipeline from start to finish 
*(data  download &rarr; data preprocessing &rarr; model traning &rarr; model evalutaion)*:

1. Navigate to the root directory:

```sh
cd thesis-rough
```

2. Run the pipeline script:

```sh
python -m run_pipeline
```

*Note:* 
A default timestamp `timestamp_out` (formatted 'MMDDYYYY_HHMM') is generated at runtime for data version control and is used consistently through pipeline. This timestamp version can be specified manually using the additional command line argument:
 ```sh
 python -m run_pipeline --timestamp_out <MMDDYYYY_HHMM>
 ```

3. View plots generated for post-training model evaluation at:
``` 
plots/quick-eval/<model_name>/<hemisphere>/<timestamp>
```

### Option B: Run Partial Pipeline

Series of steps can be run from any starting point to any stopping point specified by the user as a command line argument, as long as there are pre-existing input data available for the chosen steps:

1. Navigate to the root directory:

```sh
cd thesis-rough
```

2. Run the pipeline script with command line arguments:

```sh
python -m run_pipeline --start <step_name> --stop <step_name> --timestamp_out <MMDDYYYY_HHMM>
```

*Note:* 
If either `--start` or `--stop` point arguments are omttited, the pipeline will run from the beggining to the specified end-point or from the specified start-point to the end of the pipeline. 

**Step Names:**

- `download_ice_vel`, `download_wind`, or `download_ice_conc` Download raw ice velocity, wind, or ice concntration datasets.
- `regrid_ice_vel`, `regrid_wind`, or `regrid_ice_conc` Project raw ice velocity, wind, or ice concentration to common grid with parameter bounds.
- `mask_normalize` Masks invalid data points and normalizes model inputs.
- `process_inputs` Processes model inputs into train, test, and validation splits.
- `ps`, `lr`, `lr_wtd`, `cnn`, or `cnn_wtd` Train and evaluate persistence, linear regression, weighted linear regression, CNN, or weighted CNN models.

*Note:* 
A timestamp version, with format 'MMDDYYYY_HHMM', must be assigned to specify pre-existing source data if the pipeline is run after the raw data download steps. `timestamp_out` is the default command used to cover all steps in the pipeline. The user, however, can specify a version of step-specific data using the arguments:

- `--timestamp_raw` Specifies version of raw data
- `--timestamp_regrid` Specifies version of regrid data  
- `--timestamp_mask_norm` Specifies version of masked and normalized data
- `--timestamp_model_input` Specifies version of training, validation, and testing split inputs
- `--timestamp_model_output` Specifies version of model output


### Option C: Run Single Module

The user has the ability to run a single script as a python module for debugging or experimentation:

1. Navigate to the root directory:

```sh
cd thesis-rough
```

2. Run the pipeline script:

*Note:* `--timestamp_out` or a step-specific timestamp version must be specified for steps other than raw data download.

```sh
python -m <script_directory>.<script_name>
```

_For more examples, please refer to the [Documentation](https://example.com)_

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- ROADMAP -->
## Roadmap

- [ ] Concicse pipeline logging.
    - [ ] Data parameters, normalization, etc.
    - [ ] Global evaluation metrics
    - [ ] Memory usage and model train/ eval time
    
- [ ] Improve memory usage, (np.memmap, or torch Datasets)
- [ ] Allow GPU flexibility

- [ ] Experiment with other datasets.
- [ ] Cronjob for daily forecasts operationally.

See the [open issues](https://github.com/jackbassham/thesis-rough/issues) for a full list of proposed features (and known issues).

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- CONTRIBUTING -->
## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".
Don't forget to give the project a star! Thanks again!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Top contributors:

<a href="https://github.com/jackbassham/thesis-rough/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=jackbassham/thesis-rough" alt="contrib.rocks image" />
</a>



<!-- LICENSE -->
## License

Distributed under the project_license. See `LICENSE.txt` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- CONTACT -->
## Contact

Jack Bassham - jbassham@ucsd.edu

Project Link: [https://github.com/jackbassham/thesis-rough](https://github.com/jackbassham/thesis-rough)

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

This open source software builds on the methodology and findings presented in:

* Hoffman, L., et al. (2023). *Machine Learning for Daily Forecasts of Arctic Sea Ice Motion: An Attribution Assessment of Model Predictive Skill*.  
  *Artificial Intelligence for the Earth Systems*. https://doi.org/10.1175/AIES-D-23-0004.1
*  Zhai, J., and C. M. Bitz (2021). *A machine learning model of Arctic sea ice motions*.  
  arXiv:2108.10925. https://arxiv.org/abs/2108.10925


## Datasets

### Sea Ice Velocity
Tschudi, M., Meier, W. N., Stewart, J. S., Fowler, C. & Maslanik, J. (2019). *Polar Pathfinder Daily 25 km EASE-Grid Sea Ice Motion Vectors. (NSIDC-0116, Version 4)*. Boulder, Colorado USA. NASA National Snow and Ice Data Center Distributed Active Archive Center. https://doi.org/10.5067/INAWUWO7QH7B. Date Accessed 04-06-2026.


### Wind
Hersbach, H., Bell, B., Berrisford, P., Biavati, G., Horányi, A., Muñoz Sabater, J., Nicolas, J., Peubey, C., Radu, R., Rozum, I., Schepers, D., Simmons, A., Soci, C., Dee, D., Thépaut, J-N. (2023): ERA5 hourly data on single levels from 1940 to present. Copernicus Climate Change Service (C3S) Climate Data Store (CDS), DOI: 10.24381/cds.adbb2d47. https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels?tab=overview.  Date Accessed 04-20-2026.


### Sea Ice Concentration
DiGirolamo, N., Parkinson, C. L., Cavalieri, D. J., Gloersen, P. & Zwally, H. J. (2022). *Sea Ice Concentrations from Nimbus-7 SMMR and DMSP SSM/I-SSMIS Passive Microwave Data. (NSIDC-0051, Version 2)*. Boulder, Colorado USA. NASA National Snow and Ice Data Center Distributed Active Archive Center. https://doi.org/10.5067/MPYG15WAA4WX. Date Accessed 04-06-2026.

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/jackbassham/thesis-rough.svg?style=for-the-badge
[contributors-url]: https://github.com/jackbassham/thesis-rough/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/jackbassham/thesis-rough.svg?style=for-the-badge
[forks-url]: https://github.com/jackbassham/thesis-rough/network/members
[stars-shield]: https://img.shields.io/github/stars/jackbassham/thesis-rough.svg?style=for-the-badge
[stars-url]: https://github.com/jackbassham/thesis-rough/stargazers
[issues-shield]: https://img.shields.io/github/issues/jackbassham/thesis-rough.svg?style=for-the-badge
[issues-url]: https://github.com/jackbassham/thesis-rough/issues
[license-shield]: https://img.shields.io/github/license/jackbassham/thesis-rough.svg?style=for-the-badge
[license-url]: https://github.com/jackbassham/thesis-rough/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/jackbassham
[product-screenshot]: images/screenshot.png
<!-- Shields.io badges. You can a comprehensive list with many more badges at: https://github.com/inttter/md-badges -->
[Next.js]: https://img.shields.io/badge/next.js-000000?style=for-the-badge&logo=nextdotjs&logoColor=white
[Next-url]: https://nextjs.org/
[React.js]: https://img.shields.io/badge/React-20232A?style=for-the-badge&logo=react&logoColor=61DAFB
[React-url]: https://reactjs.org/
[Vue.js]: https://img.shields.io/badge/Vue.js-35495E?style=for-the-badge&logo=vuedotjs&logoColor=4FC08D
[Vue-url]: https://vuejs.org/
[Angular.io]: https://img.shields.io/badge/Angular-DD0031?style=for-the-badge&logo=angular&logoColor=white
[Angular-url]: https://angular.io/
[Svelte.dev]: https://img.shields.io/badge/Svelte-4A4A55?style=for-the-badge&logo=svelte&logoColor=FF3E00
[Svelte-url]: https://svelte.dev/
[Laravel.com]: https://img.shields.io/badge/Laravel-FF2D20?style=for-the-badge&logo=laravel&logoColor=white
[Laravel-url]: https://laravel.com
[Bootstrap.com]: https://img.shields.io/badge/Bootstrap-563D7C?style=for-the-badge&logo=bootstrap&logoColor=white
[Bootstrap-url]: https://getbootstrap.com
[JQuery.com]: https://img.shields.io/badge/jQuery-0769AD?style=for-the-badge&logo=jquery&logoColor=white
[JQuery-url]: https://jquery.com 
