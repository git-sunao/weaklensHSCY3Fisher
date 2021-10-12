import numpy as np
try:
    from dark_emulator_public import dark_emulator
except:
    try:
        import dark_emulator
    except:
        print('dark emulator is not installed.')
        print('See https://dark-emulator.readthedocs.io/en/latest/')
print('using dark_emulator at ', dark_emulator.__file__)
import os, sys, json, copy
import matplotlib.pyplot as plt
from collections import OrderedDict as od
try:
    from pyhalofit import pyhalofit
except:
    try:
        import pyhalofit
    except:
        print('pyhalofit is not installed.')
        print('Clone https://github.com/git-sunao/pyhalofit.')
from scipy.interpolate import InterpolatedUnivariateSpline as ius
from scipy.interpolate import interp2d, interp1d
from time import time
from scipy.integrate import simps
from tqdm import tqdm
from scipy.special import jn
from twobessel import two_Bessel
from twobessel import log_extrap
from astropy import constants

#H0 = 100 / constants.c.value *1e6 # / (Mpc/h)
H0 = 1e5 / constants.c.value # /(Mpc/h)
deg2rad = np.pi/180.0
arcmin2rad = 1/60.0 * deg2rad
arcsec2rad = 1/60.0 * arcmin2rad

#################################################
#
# Utils
#
class Silent:
    def __init__(self, silent=True):
        self.silent = silent
    def __enter__(self):
        self._original_stdout = sys.stdout
        if self.silent:
            sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.silent:
            sys.stdout.close()
        sys.stdout = self._original_stdout
        
def grep(listdir,string):
    a = list()
    for l in listdir:
        if string in l:
            a.append(l)
    return a
        
class Time:
    def __init__(self, echo=True, message=None):
        self.echo = echo
        if message is not None:
            self.message = message
        else:
            self.message = ''
        
    def __enter__(self):
        self.t_start = time()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.echo:
            print(f'{self.message}:{time()-self.t_start} sec')

def loginterp1d(x, y, xin):
    return 10**interp1d(np.log10(x), np.log10(y),
                        bounds_error=False, 
                        fill_value='extrapolate')(np.log10(xin))

def J0_binave(k, rmin, rmax):
    def i(kr):
        return kr*jn(1,kr)
    krmin, krmax = k*rmin, k*rmax
    norm = (krmax**2-krmin**2)/2.0
    ans = np.ones(k.shape)
    sel = k > 0
    ans[sel] = (i(krmax) - i(krmin))[sel]/norm[sel]
    return ans

def J2_binave(k, rmin, rmax):
    def i(kr):
        return -2*jn(0,kr) - kr*jn(1,kr)
    krmin, krmax = k*rmin, k*rmax
    norm = (krmax**2-krmin**2)/2.0
    ans = np.zeros(k.shape)
    sel = k > 0
    ans[sel] = (i(krmax) - i(krmin))[sel]/norm[sel]
    return ans

def J4_binave(k, rmin, rmax):
    def i(kr):
        return (-8/kr+kr)*jn(1,kr) - 8*jn(2, kr)
    krmin, krmax = k*rmin, k*rmax
    norm = (krmax**2-krmin**2)/2.0
    ans = np.zeros(k.shape)
    sel = k > 0
    ans[sel] = (i(krmax) - i(krmin))[sel]/norm[sel]
    return ans

def jn_binave(mu, k, rmin, rmax):
    if mu == 0.0:
        return J0_binave(k, rmin, rmax)
    elif mu == 2.0:
        return J2_binave(k, rmin, rmax)
    elif mu == 4.0:
        return J4_binave(k, rmin, rmax)

#######################################
#
# power spectrum class
#
class power_b1_class(dark_emulator.darkemu.base_class):
    def __init__(self):
        super().__init__()
        self.halofit = pyhalofit.halofit()
        self.cosmo_dict = {'omega_b': 0.02225, 'omega_c': 0.1198, 'Omega_de': 0.6844, 'ln10p10As': 3.094, 'n_s': 0.9645, 'w_de': -1.0}
        self.set_cosmology_from_dict(self.cosmo_dict)
        self.pkhalo_lchi_method = 'kz_table' # or 'lchi_table'
        
    def set_cosmology_from_dict(self, cosmo_dict):
        c = np.array([[cosmo_dict[n] for n in ['omega_b', 'omega_c', 'Omega_de', 'ln10p10As', 'n_s', 'w_de']]])
        self.set_cosmology(c)
        omn = 0.00064
        h = ((c[0][0]+c[0][1]+omn)/(1-c[0][2]))**0.5
        c = {'Omega_de0': cosmo_dict['Omega_de'], 'Omega_K0': 0.0, 'w0': cosmo_dict['w_de'], 'wa': 0.0, 'h': h}
        self.halofit.set_cosmology(c)
        cosmo_dict.update(c)
        self.cosmo_dict = cosmo_dict
    
    def get_cosmo_dict(self):
        return self.cosmo_dict
    
    def get_Dgrowth(self, z):
        _z = np.linspace(0.0, np.max(z), 20)
        _Dp = np.array([self.Dgrowth_from_z(__z) for __z in _z])
        return ius(_z, _Dp, ext=1)(z)
    
    def init_pklin(self, zmax, kbin=50, zbin=100):
        z = np.linspace(0.0, zmax, zbin)
        Dp = self.get_Dgrowth(z)
        k = np.logspace(-4, 1, kbin)
        pkL = self.get_pklin(k)
        self.pklin_data = {'z':z, 'Dp':Dp, 'k':k, 'pkL':pkL}
        self.zmax = zmax
        self.chi_max = self.get_chi_from_z(self.zmax)
        
    def init_pkhalo(self, kbin=200):
        klin = np.logspace(-4, 4, 1000)
        pkL  = self.get_pklin(klin)
        self.k_peak = klin[np.argmax(pkL)]
        z, Dp = self.pklin_data['z'], self.pklin_data['Dp']
        khalo = np.logspace(-4, 2, kbin)
        pkhalo2d = []
        for _z, _Dp in zip(z, Dp):
            pklin = pkL * _Dp**2
            self.halofit.set_pklin(klin, pklin, _z)
            pkhalo = self.halofit.get_pkhalo()
            pkhalo = 10**ius(np.log10(klin), np.log10(pkhalo))(np.log10(khalo))
            pkhalo2d.append(pkhalo)
        self.pkhalo_data = {'z':z, 'k':khalo, 'pkhalo':np.array(pkhalo2d)}
        
        if self.pkhalo_lchi_method == 'lchi_table':
            self.init_pkhalo_lchi()
        
    def init_pkhalo_lchi(self, lmin=1e-4, lmax=1e4, lbin=100, chibin=400):
        l = np.logspace(np.log10(lmin), np.log10(lmax), lbin)
        pkhalo_lchi, chi, chi_max = [], [], self.chi_max
        for _l in l:
            chi_peak = _l/self.k_peak
            if chi_peak*1e3 < chi_max:
                _chi = np.logspace(np.log10(chi_peak)-3, np.log10(chi_peak)+3, chibin)
            else:
                _chi = np.logspace(np.log10(chi_max)-3 , np.log10(chi_max), chibin)
            k = _l/_chi
            z = self.get_z_from_chi(_chi)
            _pkhalo = []
            for _k, _z in zip(k, z):
                _pkhalo.append(self.get_pkhalo_kz(_k, _z))
            pkhalo_lchi.append(_pkhalo)
            chi.append(_chi)
        pkhalo_lchi, chi = np.array(pkhalo_lchi), np.array(chi)
        self.pkhalo_lchi_data = {'l':l, 'chi':chi, 'pkhalo':pkhalo_lchi}
        
    def get_pklin_kz(self, k, z):
        return self.get_pklin_from_z(k, z)
    
    def get_pkhalo_kz(self, k, z):
        if z <= self.pkhalo_data['z'].min():
            pkhalo = loginterp1d(self.pkhalo_data['k'], self.pkhalo_data['pkhalo'][0, :], k)
            return pkhalo
        elif z>= self.pkhalo_data['z'].max():
            pkhalo = loginterp1d(self.pkhalo_data['k'], self.pkhalo_data['pkhalo'][-1, :], k)
            return pkhalo
        else:
            idx0 = np.argwhere(self.pkhalo_data['z'] < z)[-1][0]
            idx1 = idx0+1
            pkhalo0 = loginterp1d(self.pkhalo_data['k'],self.pkhalo_data['pkhalo'][idx0,:], k)
            pkhalo1 = loginterp1d(self.pkhalo_data['k'],self.pkhalo_data['pkhalo'][idx1,:], k)
            z0 = self.pkhalo_data['z'][idx0]
            z1 = self.pkhalo_data['z'][idx1]
            pkhalo = (np.log10(pkhalo1)-np.log10(pkhalo0))/(z1-z0)*(z-z0) + np.log10(pkhalo0)
            return 10**pkhalo
    
    def get_chi_from_z(self, z):
        """
        return comoving distance, chi, in Mpc/h
        """
        return self.halofit.cosmo.comoving_distance(z).value * self.cosmo_dict['h']
    
    def get_z_from_chi(self, chi):
        _z = np.linspace(0.0, 10.0, 100)
        _chi = self.get_chi_from_z(_z)
        return ius(_chi, _z)(chi)
    
    def z2chi(self, z):
        return self.get_chi_from_z(z)
    
    def chi2z(self, chi):
        return self.get_z_from_chi(chi)
    
    def get_pklin_lchi(self, l, chi):
        """
        return pk(k=l/chi, z=z(chi))
        """
        z = self.get_z_from_chi(chi)
        Dp = ius(self.pklin_data['z'], self.pklin_data['Dp'], ext=1)(z)
        k = l/chi
        pkL = loginterp1d(self.pklin_data['k'], self.pklin_data['pkL'],k)
        return Dp**2*pkL
    
    def get_pkhalo_lchi(self, l, chi):
        if self.pkhalo_lchi_method == 'lchi_table':
            return self.get_pkhalo_lchi_from_lchi_table(l, chi)
        elif self.pkhalo_lchi_method == 'kz_table':
            return self.get_pkhalo_lchi_from_kz(l, chi)
    
    def get_pkhalo_lchi_from_lchi_table(self, l, chi):
        """
        This function use a prepared table.
        However making table (self.init_pkhalo_lchi) take 
        too much time (~8sec) and this function itself is not so fast.
        naive implimentation would not be so bad.
        """
        if l <= self.pkhalo_lchi_data['l'].min():
            pkhalo = loginterp1d(self.pkhalo_lchi_data['chi'][0,:], self.pkhalo_lchi_data['pkhalo'][0, :], chi)
            return pkhalo
        elif l>= self.pkhalo_lchi_data['l'].max():
            pkhalo = loginterp1d(self.pkhalo_lchi_data['chi'][-1,:], self.pkhalo_lchi_data['pkhalo'][-1, :], chi)
            return pkhalo
        else:
            idx0 = np.argwhere(self.pkhalo_lchi_data['l'] < l)[-1][0]
            idx1 = idx0+1
            pkhalo0 = loginterp1d(self.pkhalo_lchi_data['chi'][idx0,:], self.pkhalo_lchi_data['pkhalo'][idx0,:], chi)
            pkhalo1 = loginterp1d(self.pkhalo_lchi_data['chi'][idx1,:],self.pkhalo_lchi_data['pkhalo'][idx1,:], chi)
            l0 = self.pkhalo_lchi_data['l'][idx0]
            l1 = self.pkhalo_lchi_data['l'][idx1]
            pkhalo = (np.log10(pkhalo1)-np.log10(pkhalo0))/(l1-l0)*(l-l0) + np.log10(pkhalo0)
            return 10**pkhalo
    
    def get_pkhalo_lchi_from_kz(self, l, chi):
        def sparse(chi):
            k, z = l/chi, self.chi2z(chi)
            ans = []
            for _k, _z in zip(k, z):
                ans.append(self.get_pkhalo_kz(_k, _z))
            return np.array(ans)
        if len(chi) > 100:
            chi_sparse = chi[::4]
            return loginterp1d(chi_sparse, sparse(chi_sparse),chi)
        else:
            return sparse(chi)
    
    def get_pklingm_lchi(self, l, chi, b1):
        return b1*self.get_pklin_lchi(l, chi)
    
    def get_pklingg_lchi(self, l, chi, b1_1, b1_2):
        return b1_1*b1_2*self.get_pklin_lchi(l, chi)
    
    def get_pkhalogm_lchi(self, l, chi, b1):
        return b1*self.get_pkhalo_lchi(l, chi)
    
    def get_pkhalogg_lchi(self, l, chi, b1_1, b1_2):
        return b1_1*b1_2*self.get_pkhalo_lchi(l, chi)
    
    def get_pk_lchi(self, l, chi, model):
        if model == 'lin':
            return self.get_pklin_lchi(l, chi)
        elif model  == 'nonlin':
            return self.get_pkhalo_lchi(l, chi)
        
    def get_pkmm_lchi(self, l, chi, model):
        return self.get_pk_lchi(l, chi, model)
    
    def get_pkgm_lchi(self, l, chi, b1, model):
        if model == 'lin':
            return self.get_pklingm_lchi(l, chi, b1)
        elif model  == 'nonlin':
            return self.get_pkhalogm_lchi(l, chi, b1)
        
    def get_pkgg_lchi(self, l, chi, b1_1, b1_2, model):
        if model == 'lin':
            return self.get_pklingg_lchi(l, chi, b1_1, b1_2)
        elif model  == 'nonlin':
            return self.get_pkhalogg_lchi(l, chi,  b1_1, b1_2)


#######################################
#
# windows
#
def window_lensing(chi, z, z_source, cosmo_dict, z2chi):
    ans = np.zeros(chi.shape)
    chi_source = z2chi(z_source)
    sel = chi < chi_source
    Omega_m = 1.0 - cosmo_dict['Omega_de']
    ans[sel] = 3.0/2.0 * H0**2 * Omega_m / (1+z[sel]) * chi[sel]*(chi_source-chi[sel])/chi_source
    return ans

def window_lensing_chirange(z_source, z2chi, r=0.01):
    chi_source = z2chi(z_source)
    return np.array([r*chi_source, (1-r)*chi_source])

def window_tophat(chi, z, zmin, zmax, z2chi):
    chi_max, chi_min = z2chi(zmax), z2chi(zmin)
    ans = np.zeros(chi.shape)
    sel = np.logical_and(chi_min<chi, chi<chi_max)
    dchi = chi_max-chi_min
    ans[sel] = 1/dchi
    return ans

def window_tophat_chirange(zmin, zmax, z2chi):
    chi_max, chi_min = z2chi(zmax), z2chi(zmin)
    return np.array([chi_min, chi_max])

###########################################
#
# making chi array for a given window pair
#
def get_chirange_overlap(chirange1, chirange2):
    if chirange1.max() <= chirange2.min():
        # no overlap
        return np.array([0.0, 0.0])
    elif chirange2.max() <= chirange1.min():
        # no overlap
        return np.array([0.0, 0.0])
    else:
        chi_min, chi_max = max([chirange1.min(), chirange2.min()]), min([chirange1.max(), chirange2.max()])
        return np.array([chi_min, chi_max])
    
def get_chi_overlap(chirange1, chirange2, chi_bin, return_int_flag=False):
    cr = get_chirange_overlap(chirange1, chirange2)
    if return_int_flag:
        if np.all(cr>0):
            return np.linspace(cr[0], cr[1], chi_bin), True
        else:
            return np.linspace(cr[0], cr[1], chi_bin), False
    else:
        return np.linspace(cr[0], cr[1], chi_bin)

def get_chi_lensing(chirange1, chirange2, chi_bin, chi_min_max):
    cr = get_chirange_overlap(chirange1, chirange2)
    if chi_min_max > cr[0]:
        #print('lin scale is enough for lensing kernel.')
        chi = np.linspace(cr[0], cr[1], chi_bin)
    else:
        #print('log scale for lensing kernel.')
        chi = np.logspace(np.log10(chi_min_max), np.log10(cr[1]), chi_bin)
    return chi

#######################################
#
# galaxy sample classes
#
class galaxy_base_class:
    info_keys = []
    sample_type = ''
    def __init__(self, info):
        self.set_galaxy_info(info)
        self.halofit = pyhalofit.halofit()
        
    def set_galaxy_info(self, info):
        if isinstance(info, dict):
            self._set_galaxy_info_from_dict(info)
        elif isinstance(info, list):
            self._set_galaxy_info_from_list(info)
        
    def _set_galaxy_info_from_list(self, info_list):
        self._set_galaxy_info_from_dict(od(zip(self.info_keys, info_list)))
    
    def _set_galaxy_info_from_dict(self, info):
        self.info = info
    
    def get_sample_info(self):
        return self.info
    
    def set_cosmology_from_dict(self, cosmo_dict):
        self.halofit.set_cosmology(cosmo_dict)
        self.cosmo_dict = cosmo_dict
        
    def z2chi(self, z):
        return self.halofit.cosmo.comoving_distance(z).value * self.cosmo_dict['h']
    
    def chi2z(self, chi):
        _z = np.linspace(0.0, 10.0, 100)
        _chi = self.z2chi(_z)
        return ius(_chi, _z)(chi)
    
    def get_shot_noise(self):
        v, unit = self.info['n2d'].split(' ')[:2]
        if 'arcmin^-2' in unit:
            ns = float(v) / arcmin2rad**2
        elif 'rad^-2' in unit:
            ns = float(v)
        elif 'deg^-2' in unit:
            ns = float(v)/deg2rad**2
        elif 'arcsec^-2' in unit:
            ns = float(v)/arcsec2rad**2
        return 1/ns
    
    def dump(self, dirname):
        fname = 'galaxy_sample_' + self.info['sample_name'] + '.json'
        print(f'saved galaxy sample {self.info["sample_name"]} to {fname}')
        info = copy.deepcopy(self.info)
        info['sample_type'] = self.sample_type
        json.dump(info, open(os.path.join(dirname, fname) ,'w'), indent=2)
        

class galaxy_sample_source_class(galaxy_base_class):
    info_keys = ['sample_name', 'Pzs_fname', 'dzph', 'dm', 'sigma_shape', 'n2d']
    sample_type = 'source'
    def __init__(self, info):
        super().__init__(info)
        
    def set_cosmology_from_dict(self, cosmo_dict):
        super().set_cosmology_from_dict(cosmo_dict)
        # chi_s
        # 
        # \int{\rm d}z_l P(z_l) \chi (\chi_s-\chi)/\chi_s = \chi (1 - \chi <\chi_s^{-1}> )
        #
        # where <\chi_s^{-1}> = \int{\rm d}z_s P(z_s) \chi_s^{-1}.
        #
        # We define z_source_eff as 
        #
        #  \chi(z_source_eff) = <\chi_s^{-1}>^{-1}
        zs, Pzs = np.loadtxt(self.info['Pzs_fname'], unpack=True)
        z = np.linspace(zs.min(), zs.max(), 100)
        norm = simps(ius(zs, Pzs, ext=1)(z), z)
        chi_s = self.z2chi(z)
        chi_s_inv_ave = simps(ius(zs, Pzs, ext=1)(z+self.info['dzph']) * 1/chi_s, z)/norm # = <\chi_s^{-1}>
        self.z_source_eff = self.chi2z(chi_s_inv_ave**-1)
        self.cosmo_dict = cosmo_dict
    
    def window_lensing(self, chi, z):
        return window_lensing(chi, z, self.z_source_eff, self.cosmo_dict, self.z2chi)
    
    def window_lensing_chirange(self):
        return window_lensing_chirange(self.z_source_eff, self.z2chi)
    
    def get_chi_source(self):
        return self.z2chi(self.z_source_eff)
    
    def get_shot_noise(self):
        v = super().get_shot_noise()
        return self.info['sigma_shape']**2*v
    
class galaxy_sample_lens_class(galaxy_base_class):
    info_keys = ['sample_name', 'z_min', 'z_max', 'galaxy_bias', 'n2d', 'alpha_mag']
    sample_type = 'lens'
    def __init__(self, info):
        super().__init__(info)
        
    def set_cosmology_from_dict(self, cosmo_dict):
        super().set_cosmology_from_dict(cosmo_dict)
        # chi_lens
        # 
        # \int{\rm d}z_l P(z_l) \chi (\chi_l-\chi)/\chi_l = \chi (<\chi_l^{-1}>^{-1} - \chi) / <\chi_l^{-1}>^{-1}
        #
        # where <\chi_l^{-1}> = \int{\rm d}z_l P(z_l) \chi_l^{-1}.
        #
        # We define z_lens_eff as 
        #
        #  \chi(z_lens_eff) = <\chi_l^{-1}>^{-1}
        # 
        # We assume P(z_l) = {\rm const} for simplicity.
        z = np.linspace(self.info['z_min'], self.info['z_max'], 100)
        chi = self.z2chi(z)
        chi_lens_inv_ave = simps(1/chi, z)/(self.info['z_max']-self.info['z_min']) # = <\chi_l^{-1}>
        self.z_lens_eff = self.chi2z(chi_lens_inv_ave**-1)
        self.cosmo_dict = cosmo_dict
    
    def window_galaxy(self, chi, z):
        return window_tophat(chi, z, self.info['z_min'], self.info['z_max'], self.z2chi)
    
    def window_galaxy_chirange(self):
        return window_tophat_chirange(self.info['z_min'], self.info['z_max'], self.z2chi)
    
    def window_magnification(self, chi, z):
        return window_lensing(chi, z, self.z_lens_eff, self.cosmo_dict, self.z2chi)
    
    def window_magnification_chirange(self):
        return window_lensing_chirange(self.z_lens_eff, self.z2chi)
    
    def get_chi_lens(self):
        return self.z2chi(self.z_lens_eff)
    
    def get_Delta_chi(self):
        cr = self.window_galaxy_chirange()
        return cr.max()-cr.min()
    
    def get_Delta_chi_g(self):
        return self.get_Delta_chi()

#######################################
#
# pk to cl class
#
class pk2cl_class:
    """
    This class compute a convolution like
    ::math::
        C(l) = P(l) = \int{\rm d}\chi W_1(\chi )W_2(\chi)P\left(\frac{l}{\chi}, z(\chi)\right)
    """
    probe_mu_dict = {'xi+':0, 'xi-':4, 'w':0, 'gamma_t':2}
    probe_latex_dict = {'xi+':r'$\xi_+$', 'xi-':r'$\xi_-$', 'gamma_t':r'$\gamma_\mathrm{t}$', 'w':r'$w$'}
    def __init__(self, pk_class=None):
        self.pk_class = pk_class
        self.galaxy_sample_dict = od()
        self.cl_cache = od()
        self.cl_cache_sep = '-'
        self.chi_bin_lens_kernel = 50 # used to compute \int \diff\chi W_lensing(\chi)W_lensing(\chi)/\chi^2 P(l/\chi, z(\chi))
        self.chi_bin_galaxy_window = 20 # used to compute \int\diff\chi W_galaxy(\chi)W_X(\chi) /\chi^2 P(l/\chi, z(\chi))
        self.Cl_names_sep = ','
        
    def set_galaxy_sample(self, galaxy_sample):
        if galaxy_sample.info['sample_name'] in self.galaxy_sample_dict.keys():
            print(f'Note: Sample name {galaxy_sample.info["sample_name"]} is already used.\nPreviously registered sample is replaced.')
        self.galaxy_sample_dict[galaxy_sample.info['sample_name']] = galaxy_sample
        
    def get_galaxy_sample_names(self):
        return list(self.galaxy_sample_dict.keys())
    
    def set_Omega_s(self, Omega_s_dict):
        """
        At least, Omega_s for the keys 'w', 'gamma_t', 'xi' must be supplied.
        Unit is deg^2
        """
        self.Omega_s = Omega_s_dict
        
    def set_cosmology_from_dict(self, cosmo_dict=None):
        if cosmo_dict is not None:
            self.pk_class.set_cosmology_from_dict(cosmo_dict)
        cosmo_dict = self.pk_class.get_cosmo_dict()
        for key in self.galaxy_sample_dict.keys():
            self.galaxy_sample_dict[key].set_cosmology_from_dict(cosmo_dict)
        self.cosmo_dict = cosmo_dict
        
    def get_cosmo_dict(self):
        return copy.deepcopy(self.cosmo_dict)
        
    def get_zmax_from_galaxy_samples(self):
        zmax = 0
        for key, sample in self.galaxy_sample_dict.items():
            if sample.sample_type == 'source':
                _zmax = sample.z_source_eff
            elif sample.sample_type == 'lens':
                _zmax = sample.z_lens_eff
            if _zmax > zmax:
                zmax = _zmax
        return zmax
    
    def init_pk(self):
        zmax = self.get_zmax_from_galaxy_samples()
        self.pk_class.init_pklin(zmax)
        self.pk_class.init_pkhalo()
        self.k_peak = self.pk_class.k_peak
    
    def get_galaxy_sample(self, name):
        return self.galaxy_sample_dict[name]
        
    def _Cgg(self, sample1, sample2, l, model='nonlin', plot=False, plot_xlog=False):
        ans = 0
        b1_1, b1_2 = sample1.info['galaxy_bias'], sample2.info['galaxy_bias']
        alpha_mag1, alpha_mag2 = sample1.info['alpha_mag'], sample2.info['alpha_mag']
        # g g
        chi, int_flag = get_chi_overlap( sample1.window_galaxy_chirange(), sample2.window_galaxy_chirange(), self.chi_bin_galaxy_window, return_int_flag=True)
        if int_flag:
            z = self.pk_class.get_z_from_chi(chi)
            plchi = self.pk_class.get_pkgg_lchi(l, chi, b1_1, b1_2, model)
            w1, w2 = sample1.window_galaxy(chi, z), sample2.window_galaxy(chi, z)
            ans += simps(w1*w2*plchi/chi**2, chi)
        if plot and int_flag:
            plt.figure()
            plt.xlabel(r'$\chi$')
            plt.yscale('log')
            plt.xscale('log') if plot_xlog else None
            plt.plot(chi, w1*w2*plchi/chi**2, label='g, g')
            plt.plot(chi, w1*w2*plchi.max()/chi**2)
            print(f'Cgg(l)                                ={ans}')
        # g mag
        chi, int_flag = get_chi_overlap( sample1.window_galaxy_chirange(), sample2.window_magnification_chirange(), self.chi_bin_galaxy_window, return_int_flag=True)
        if int_flag:
            z = self.pk_class.get_z_from_chi(chi)
            plchi = self.pk_class.get_pkgm_lchi(l, chi, b1_1, model)
            w1, w2 = sample1.window_galaxy(chi, z), sample2.window_magnification(chi, z)
            ans += simps(w1*w2*plchi/chi**2, chi) * 2*(alpha_mag2-1)
        if plot and int_flag:
            plt.plot(chi, w1*w2*plchi/chi**2 * 2*(alpha_mag2-1), label='g, mag')
            print(f'Cgg(l)+Cg,mag(l)                      ={ans}')
        # mag g
        chi, int_flag = get_chi_overlap( sample1.window_magnification_chirange(), sample2.window_galaxy_chirange(), self.chi_bin_galaxy_window, return_int_flag=True)
        if int_flag:
            z = self.pk_class.get_z_from_chi(chi)
            plchi = self.pk_class.get_pkgm_lchi(l, chi, b1_2, model)
            w1, w2 = sample1.window_magnification(chi, z), sample2.window_galaxy(chi, z)
            ans += simps(w1*w2*plchi/chi**2, chi) * 2*(alpha_mag1-1)
        if plot and int_flag:
            plt.plot(chi, w1*w2*plchi/chi**2 * 2*(alpha_mag1-1), label='mag, g')
            print(f'Cgg(l)+Cg,mag(l)+Cmag,g(l)            ={ans}')
        # mag mag
        chi = get_chi_lensing( sample1.window_magnification_chirange(), sample2.window_magnification_chirange(), 
                              self.chi_bin_lens_kernel, l/self.k_peak/100.0)
        z = self.pk_class.get_z_from_chi(chi)
        plchi = self.pk_class.get_pkmm_lchi(l, chi, model)
        w1, w2 = sample1.window_magnification(chi, z), sample2.window_magnification(chi, z)
        ans += simps(w1*w2*plchi/chi**2, chi) * 2*(alpha_mag1-1) * 2*(alpha_mag2-1)
        if plot:
            plt.plot(chi, w1*w2*plchi/chi**2 * 2*(alpha_mag1-1) * 2*(alpha_mag2-1), label='mag,mag')
            plt.legend()
            print(f'Cgg(l)+Cg,mag(l)+Cmag,g(l)+Cmag,mag(l)={ans}')
        return ans

    def _CgE(self, sample_l, sample_s, l, model='nonlin',plot=False, plot_xlog=False):
        ans = 0
        b1, alpha = sample_l.info['galaxy_bias'], sample_l.info['alpha_mag']
        m = 1.0 + sample_s.info['dm']
        # g E
        chi, int_flag = get_chi_overlap( sample_l.window_galaxy_chirange(), sample_s.window_lensing_chirange(), self.chi_bin_galaxy_window, return_int_flag=True)
        if int_flag:
            z = self.pk_class.get_z_from_chi(chi)
            plchi = self.pk_class.get_pkgm_lchi(l, chi, b1, model)
            w1, w2 = sample_l.window_galaxy(chi, z), sample_s.window_lensing(chi, z)
            ans += simps(w1*w2*plchi/chi**2, chi)*m
        if plot and int_flag:
            plt.figure()
            plt.yscale('log')
            plt.xscale('log') if plot_xlog else None
            plt.plot(chi, w1*w2*plchi/chi**2)
        # mag E
        chi = get_chi_lensing( sample_l.window_magnification_chirange(), sample_s.window_lensing_chirange(), 
                              self.chi_bin_lens_kernel, l/self.k_peak/100.0)
        z = self.pk_class.get_z_from_chi(chi)
        plchi = self.pk_class.get_pkmm_lchi(l, chi, model)
        w1, w2 = sample_l.window_magnification(chi, z), sample_s.window_lensing(chi, z)
        ans += simps(w1*w2*plchi/chi**2, chi) * 2*(alpha-1)*m
        if plot:
            plt.plot(chi, w1*w2*plchi/chi**2 * 2*(alpha-1))
        return ans
    
    def _CEE(self, sample1, sample2, l, model='nonlin', plot=False, plot_xlog=False):
        # EE
        m1, m2 = 1.0+sample1.info['dm'], 1.0+sample2.info['dm']
        chi = get_chi_lensing( sample1.window_lensing_chirange(), sample2.window_lensing_chirange(), 
                              self.chi_bin_lens_kernel, l/self.k_peak/100.0)
        z = self.pk_class.get_z_from_chi(chi)
        plchi = self.pk_class.get_pkmm_lchi(l, chi, model)
        w1, w2 = sample1.window_lensing(chi, z), sample2.window_lensing(chi, z)
        ans = simps(w1*w2*plchi/chi**2, chi)*m1*m2
        if plot:
            plt.figure()
            plt.yscale('log')
            plt.xscale('log') if plot_xlog else None
            plt.plot(chi, w1*w2*plchi/chi**2)
        return ans
    
    def _logspace_l_array(self, min_max_bin_list):
        l = []
        for args in min_max_bin_list:
            log_lmin, log_lmax, lbin = args
            _l = np.logspace(log_lmin, log_lmax, lbin)
            l.append(_l)
        l = np.hstack(l)
        return sorted(l)
    
    def _l_array_peakBAO(self, llp):
        l_sparse = self._logspace_l_array([[llp-4  , llp-0.5, 10],
                                          [llp-0.5, llp+1.4, 50], # peak & BAO
                                          [llp+1.4, llp+4.0, 10 ]])
        return l_sparse
    
    def Cgg(self, name1, name2, l, model='nonlin', plot=False):
        # get samples
        sample1, sample2 = self.galaxy_sample_dict[name1], self.galaxy_sample_dict[name2]
        # init l array for interplation
        chi_mean = (sample1.get_chi_lens()+sample2.get_chi_lens())/2
        llp = np.log10(self.pk_class.k_peak*chi_mean) # log_l_peak
        l_sparse = self._l_array_peakBAO(llp)
        ans_sparse = [ self._Cgg(sample1, sample2, _l, model=model) for _l in l_sparse]
        if plot:
            plt.figure()
            plt.loglog(l_sparse, ans_sparse, marker='.')
            plt.show()
        return loginterp1d(l_sparse, ans_sparse, l)
    
    def CgE(self, name1, name2, l, model='nonlin', plot=False):
        # get samples
        sample1, sample2 = self.galaxy_sample_dict[name1], self.galaxy_sample_dict[name2]
        if sample1.sample_type == 'lens' and sample2.sample_type == 'source':
            sample_l, sample_s = sample1, sample2
        elif sample1.sample_type == 'source' and sample2.sample_type == 'lens':
            sample_l, sample_s = sample2, sample1
        # init l array for interpolation
        llp = np.log10(self.pk_class.k_peak*sample_l.get_chi_lens()) # log_l_peak
        l_sparse = self._l_array_peakBAO(llp)
        ans_sparse = np.array([ self._CgE(sample_l, sample_s, _l, model=model) for _l in l_sparse])
        if plot:
            plt.figure()
            plt.loglog(l_sparse, ans_sparse, marker='.')
            plt.show()
        return loginterp1d(l_sparse, ans_sparse, l)
    
    def CEE(self, name1, name2, l, model='nonlin', plot=False):
        sample1, sample2 = self.galaxy_sample_dict[name1], self.galaxy_sample_dict[name2]
        l_sparse = np.logspace(-2, 4, 60)
        ans_sparse = np.array([ self._CEE(sample1, sample2, _l, model=model) for _l in l_sparse])
        return loginterp1d(l_sparse, ans_sparse, l)
    
    def get_samples(self, sample_type=''):
        lens_samples = []
        for name, sample in self.galaxy_sample_dict.items():
            if sample.sample_type == 'lens':
                lens_samples = 0
    
    def compute_all_Cl(self, l, model='nonlin', compute_lens_cross=True):
        lens_samples   = [name for name, s in self.galaxy_sample_dict.items() if s.sample_type == 'lens']
        source_samples = [name for name, s in self.galaxy_sample_dict.items() if s.sample_type == 'source']
        
        self.Cl_cache = od()
        self.Cl_cache['condition'] = od()
        self.Cl_cache['condition'] = model
        
        self.Cl_cache['Cl'] = od()
        self.Cl_cache['Cl']['l'] = l
        def helper(name1, name2, func):
            key = self.Cl_names_sep.join([name1, name2])
            key_inv = self.Cl_names_sep.join([name2, name1])
            if key_inv in self.Cl_cache.keys():
                self.Cl_cache['Cl'][key] = self.Cl_cache[key_inv]
            else:
                self.Cl_cache['Cl'][key] = func(name1, name2, l, model=model)
        # Cgg
        if compute_lens_cross:
            for name1 in lens_samples:
                for name2 in lens_samples:
                    helper(name1, name2, self.Cgg)
        else:
            for name1 in lens_samples:
                helper(name1, name1, self.Cgg)
        # CgE
        for name1 in lens_samples:
            for name2 in source_samples:
                helper(name1, name2, self.CgE)
        # CEE
        for name1 in source_samples:
            for name2 in source_samples:
                helper(name1, name2, self.CEE)
                
    def dump_Cl_cache(self,dirname, overwrite=False):
        if (not os.path.exists(dirname)) or overwrite:
            if not os.path.exists(dirname):
                os.makedirs(dirname)
            # Cl
            for key in self.Cl_cache['Cl'].keys():
                fname = os.path.join(dirname, key+'.txt')
                print(f'saving {key} to {fname}')
                np.savetxt(fname, self.Cl_cache['Cl'][key])
            # cosmo dict
            print(f'saving cosmo_dict to {dirname}/cosmo_dict.json.')
            json.dump(self.cosmo_dict, open(os.path.join(dirname, 'cosmo_dict.json'), 'w'), indent=2)
            # galaxy samples
            for k, sample in self.galaxy_sample_dict.items():
                sample.dump(dirname)
            # survey area
            json.dump(self.Omega_s, open(os.path.join(dirname, 'Omega_s.json'), 'w'), indent=2)
        else:
            print(f'{dirname} already exists.')
            
    def load_Cl_cache(self, dirname):
        self.Cl_cache = od()
        # Cl
        self.Cl_cache['Cl'] = od()
        fnames = grep(os.listdir(dirname), '.txt')
        for fname in fnames:
            key = fname.replace('.txt','')
            self.Cl_cache['Cl'][key] = np.loadtxt(os.path.join(dirname, fname))
        registered_key = list(self.Cl_cache['Cl'].keys())
        for key in registered_key:
            key_inv = self.Cl_names_sep.join(key.split(self.Cl_names_sep)[::-1])
            if not key_inv in self.Cl_cache['Cl'].keys():
                self.Cl_cache['Cl'][key_inv] = self.Cl_cache['Cl'][key]
        # galaxy samples
        fnames = grep(os.listdir(dirname), 'galaxy_sample')
        for fname in fnames:
            info = json.load(open(os.path.join(dirname, fname), 'r'), object_pairs_hook=od)
            #self._set_galaxy_info_from_dict(info)
            sample_type = info.pop('sample_type')
            if sample_type == 'lens':
                gs = galaxy_sample_lens_class(info)
            elif sample_type == 'source':
                gs = galaxy_sample_source_class(info)
            self.set_galaxy_sample(gs)
        # cosmo_dict
        cosmo_dict = json.load(open(os.path.join(dirname, 'cosmo_dict.json'), 'r'),
                               object_pairs_hook=od)
        self.set_cosmology_from_dict(cosmo_dict)
        # survey area
        Omega_s_dict = json.load(open(os.path.join(dirname, 'Omega_s.json'), 'r'), object_pairs_hook=od)
        self.set_Omega_s(Omega_s_dict)
        
        
    def get_lCl_from_cache(self, name1, name2, include_shot_noise=False):
        l = copy.deepcopy(self.Cl_cache['Cl']['l'])
        Cl = copy.deepcopy(self.Cl_cache['Cl'][ self.Cl_names_sep.join([name1, name2]) ])
        
        # more factors for wp and dSigma to be in physical scale
        #if probe == 'wp':
        #    Delta_chi_g = self.galaxy_sample_dict[name1].get_Delta_chi_g()
        #    Cl *= Delta_chi_g
        
        # shot noise term
        if include_shot_noise and name1 == name2:
            sample = self.galaxy_sample_dict[name1]
            shot_noise = sample.get_shot_noise()
            Cl += shot_noise
        return l, Cl
        
    def angular_correlation_function_fftlog(self, name1, name2, theta, probe, binave=False, dlnt=None, plot=False, plot_with=None):
        l, Cl = self.get_lCl_from_cache(name1, name2)
        
        logtmin, logtmax = np.log10(theta.min())-2, np.log10(theta.max())+2
        print(logtmin, logtmax)
        fftlog = dark_emulator.pyfftlog_interface.fftlog(logrmin=logtmin, logrmax=logtmax, num=2)
        mu = self.probe_mu_dict[probe]
        input_Cl = loginterp1d(l, Cl, fftlog.k)*fftlog.k
        
        fftlog_out = fftlog.iHT(input_Cl, mu, 0.0, -1, 1.0/2.0/np.pi)
        
        if binave:
            None # do something
        #acf = loginterp1d(fftlog.r, fftlog_out, theta)
        acf = ius(fftlog.r, fftlog_out)(theta)
        
        if plot:
            plt.figure()
            plt.xlabel(r'$\theta$')
            plt.ylabel(self.probe_latex_dict[probe])
            plt.loglog(fftlog.r, fftlog_out, c='C0', label='fftlog on fftlog grid')
            plt.loglog(fftlog.r,-fftlog_out, c='C0',ls='--')
            plt.loglog(theta, acf, c='C1', label='fftlog')
            plt.loglog(theta,-acf, c='C1', ls='--')
            if plot_with is not None:
                for d in plot_with:
                    plt.plot(d['xy'][0], d['xy'][1], c=d['c'], label=d.get('label', ''))
                    plt.plot(d['xy'][0],-d['xy'][1], c=d['c'], ls='--')
            plt.legend()
            plt.show()
        return acf
    
    def angular_correlation_function_bruteforce(self, name1, name2, theta, probe, binave=False, dlnt=None, plot=False):
        l, Cl = self.get_lCl_from_cache(name1, name2)

        if binave and dlnt is None:
            dlnt = np.log(theta[1]/theta[0])

        mu = self.probe_mu_dict[probe]

        def helper(t):
            dump = np.exp(-(l*t)**2/(50*np.pi)**2)
            if binave:
                j = jn_binave(mu, l, t, t*np.exp(dlnt))
            else:
                j = jn(mu, l*t)
            ans = simps(l**2*Cl/2.0/np.pi * j * dump, np.log(l))
            return ans

        acf = np.array([ helper(t) for t in theta])
        return acf
    
    def _get_lClCl(self, names1, probe1, names2, probe2, include_shot_noise=True):
        ClCl = 0
        if probe1 == probe2 or (probe1=='xi+' and probe2=='xi-') or (probe1=='xi-' and probe2=='xi+'):
            l, Cl1 = self.get_lCl_from_cache(names1[0], names2[0], include_shot_noise=include_shot_noise)
            l, Cl2 = self.get_lCl_from_cache(names1[1], names2[1], include_shot_noise=include_shot_noise)
            ClCl += Cl1*Cl2
            l, Cl1 = self.get_lCl_from_cache(names1[0], names2[1], include_shot_noise=include_shot_noise)
            l, Cl2 = self.get_lCl_from_cache(names1[1], names2[0], include_shot_noise=include_shot_noise)
            ClCl += Cl1*Cl2
        return l, ClCl
    
    def get_Omega_s(self, probe1, probe2):
        if probe1 == 'w' and probe2 == 'w':
            return self.Omega_s['w'] * deg2rad**2
        elif probe1 == 'gamma_t' and probe2 == 'gamma_t':
            return self.Omega_s['gamma_t'] * deg2rad**2
        elif 'xi' in probe1 and 'xi' in probe2:
            return self.Omega_s['xi'] * deg2rad**2
        else:
            print(f'No proper Omega_s is found for {probe1} and {probe2}. return 1.')
            return 4.0*np.pi
            
    def covariance_fftlog(self, names1, probe1, theta1, names2, probe2, theta2, binave=False, dlnt1=None, dlnt2=None, plot=False):
        l, ClCl = self._get_lClCl(names1, probe1, names2, probe2)
        Omega_s = self.get_Omega_s(probe1, probe2)
        dlnl = np.log(l[1]/l[0])
        input_f = np.diag(ClCl*l**2)/Omega_s*(2.0*np.pi)/dlnl / (2.0*np.pi)**2
        nu = 1.01
        tB = two_Bessel(l, l, input_f, nu1=nu, nu2=nu, 
                        N_extrap_low=0, N_extrap_high=0, c_window_width=0.25, N_pad=0)
        mu1 = self.probe_mu_dict[probe1]
        mu2 = self.probe_mu_dict[probe2]
        if binave and dlnt1 is None:
            dlnt1 = np.log(theta1[1]/theta1[0])
        if binave and dlnt2 is None:
            dlnt2 = np.log(theta2[1]/theta2[0])
        if not binave:
            dlnt1, dlnt2 = 0, 0
        print(dlnt1, dlnt2)
        if binave:
            t1_min, t2_min, cov_fine = tB.two_Bessel_binave(mu1, mu2, dlnt1, dlnt2)
        else:
            # no binave, not implemented yet.
            t1_min, t2_min, cov_fine = tB.two_Bessel_binave(mu1, mu2, dlnt1, dlnt2)
        cov = interp2d(t1_min, t2_min, cov_fine)(theta1, theta2)
        
        if plot:
            fig = plt.figure(figsize=(10, 8))
            ax1 = fig.add_subplot(2,2,1)
            im1 = ax1.imshow(cov)
            fig.colorbar(im1, ax=ax1)
            ax1.set_title(r'${\rm Cov}$['+self.probe_latex_dict[probe1]+r'$(\theta_1)$,'+
                          self.probe_latex_dict[probe2]+r'$(\theta_2)$]')

            ax2 = fig.add_subplot(2,2,2)
            ax2.loglog(t1_min, np.diag(cov_fine))
            ax2.loglog(theta1, np.diag(cov))
            ax2.set_xlabel(r'$\theta$')
            ax2.set_ylabel(r'${\rm Cov}$['+self.probe_latex_dict[probe1]+r'$(\theta)$,'+
                           self.probe_latex_dict[probe2]+r'$(\theta)$]')
            
            ax3 = fig.add_subplot(2,2,3)
            v = np.diag(cov)
            v1, v2 = np.meshgrid(v,v)
            cc = cov/(v1*v2)**0.5 # correlation coeff
            im3 = ax3.imshow(cc)
            fig.colorbar(im3, ax=ax3)
            ax3.set_title('correlation coefficients')

            plt.tight_layout()
            plt.show()
        return cov
    
    def covariance_bruteforce(self, names1, probe1, theta1, names2, probe2, theta2, binave=False, dlnt1=None, dlnt2=None, plot=False):
        l, ClCl = self._get_lClCl(names1, probe1, names2, probe2)
        Omega_s = self.get_Omega_s(probe1, probe2)
        
        if binave and dlnt1 is None:
            dlnt1 = np.log(theta1[1]/theta1[0])
        if binave and dlnt2 is None:
            dlnt2 = np.log(theta2[1]/theta2[0])
        if not binave:
            dlnt1, dlnt2 = 0, 0
        print(dlnt1, dlnt2)
        
        mu1 = self.probe_mu_dict[probe1]
        mu2 = self.probe_mu_dict[probe2]
        
        def helper(t1, t2):
            f = l**2*ClCl/2.0/np.pi/Omega_s
            dump = 1 #np.exp(-(l*t1)**2/(100*np.pi)**2 -(l*t2)**2/(100*np.pi)**2 )
            if binave:
                jj = jn_binave(mu1, l, t1, t1*np.exp(dlnt1))*jn_binave(mu2, l, t2, t2*np.exp(dlnt2))
            else:
                jj = jn(mu1, l*t1)*jn(mu2, l*t2)
            ans = simps(f*jj*dump, np.log(l))
            return ans
        
        cov = []
        for t1 in theta1:
            _cov = []
            for t2 in theta2:
                _cov.append(helper(t1, t2))
            cov.append(_cov)
        cov = np.array(cov)
        
        if plot:
            fig = plt.figure(figsize=(10, 8))
            ax1 = fig.add_subplot(2,2,1)
            im1 = ax1.imshow(cov)
            fig.colorbar(im1, ax=ax1)
            ax1.set_title(r'${\rm Cov}$['+self.probe_latex_dict[probe1]+r'$(\theta_1)$,'+
                          self.probe_latex_dict[probe2]+r'$(\theta_2)$]')

            ax2 = fig.add_subplot(2,2,2)
            ax2.loglog(theta1, np.diag(cov))
            ax2.set_xlabel(r'$\theta$')
            ax2.set_ylabel(r'${\rm Cov}$['+self.probe_latex_dict[probe1]+r'$(\theta)$,'+
                           self.probe_latex_dict[probe2]+r'$(\theta)$]')
            
            ax3 = fig.add_subplot(2,2,3)
            v = np.diag(cov)
            v1, v2 = np.meshgrid(v,v)
            cc = cov/(v1*v2)**0.5 # correlation coeff
            im3 = ax3.imshow(cc)
            fig.colorbar(im3, ax=ax3)
            ax3.set_title('correlation coefficients')

            plt.tight_layout()
            plt.show()
        
        return cov
    
      
# shape noise signa^2/n or sigma^2/2/n ?
# binave implementation
# 