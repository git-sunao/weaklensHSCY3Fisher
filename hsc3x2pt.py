import numpy as np
try:
    from dark_emulator_public import dark_emulator
except:
    import dark_emulator
print('using dark_emulator at ', dark_emulator.__file__)
import os, sys, json
import matplotlib.pyplot as plt
from collections import OrderedDict as od
try:
    from pyhalofit import pyhalofit
except:
    print('pyhalofit is not installed. Clone https://github.com/git-sunao/pyhalofit.')
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
            print(f'{self.message}:{time()-self.t_start}')

def loginterp1d(x, y, xin):
    return 10**interp1d(np.log10(x), np.log10(y),
                        bounds_error=False, 
                        fill_value='extrapolate')(np.log10(xin))

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
    
def get_chi_overlap(chirange1, chirange2, chi_bin):
    cr = get_chirange_overlap(chirange1, chirange2)
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

class galaxy_sample_source_class(galaxy_base_class):
    info_keys = ['sample_name', 'z_source', 'sigma_shape', 'n2d']
    sample_type = 'source'
    def __init__(self, info):
        super().__init__(info)
    
    def window_lensing(self, chi, z):
        return window_lensing(chi, z, self.info['z_source'], self.cosmo_dict, self.z2chi)
    
    def window_lensing_chirange(self):
        return window_lensing_chirange(self.info['z_source'], self.z2chi)
    
    def get_chi_source(self):
        return self.z2chi(self.info['z_source'])

class galaxy_sample_lens_class(galaxy_base_class):
    info_keys = ['sample_name', 'z_lens', 'z_min', 'z_max', 'galaxy_bias', 'n2d', 'alpha_mag']
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
    
    def window_galaxy(self, chi, z):
        return window_tophat(chi, z, self.info['z_min'], self.info['z_max'], self.z2chi)
    
    def window_galaxy_chirange(self):
        return window_tophat_chirange(self.info['z_min'], self.info['z_max'], self.z2chi)
    
    def window_magnification(self, chi, z):
        return window_lensing(chi, z, self.z_lens_eff, self.cosmo_dict, self.z2chi)
    
    def window_magnification_chirange(self):
        return window_lensing_chirange(self.z_lens_eff, self.z2chi)
    
    def get_chi_lens(self):
        return self.z2chi(self.info['z_lens'])
    
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
    probe_mu_dict = {'xi_plus':0, 'xi+':0, 'xi_minus':4, 'xi-':4, 'wp':0, 'gamma_t':2}
    probe_latex_dict = {'xi+':r'$\xi_+$', 'xi-':r'$\xi_-$', 'gamma_t':r'$\gamma_\mathrm{t}$', 'wp':r'$w_\mathrm{p}$'}
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
        
    def set_cosmology_from_dict(self, cosmo_dict=None):
        if cosmo_dict is not None:
            self.pk_class.set_cosmology_from_dict(cosmo_dict)
        cosmo_dict = self.pk_class.get_cosmo_dict()
        for key in self.galaxy_sample_dict.keys():
            self.galaxy_sample_dict[key].set_cosmology_from_dict(cosmo_dict)
        self.cosmo_dict = cosmo_dict
        
    def get_zmax_from_galaxy_samples(self):
        zmax = 0
        for key, sample in self.galaxy_sample_dict.items():
            if sample.sample_type == 'source':
                _zmax = sample.info['z_source']
            elif sample.sample_type == 'lens':
                _zmax = sample.info['z_lens']
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
        chi = get_chi_overlap( sample1.window_galaxy_chirange(), sample2.window_galaxy_chirange(), self.chi_bin_galaxy_window)
        z = self.pk_class.get_z_from_chi(chi)
        plchi = self.pk_class.get_pkgg_lchi(l, chi, b1_1, b1_2, model)
        w1, w2 = sample1.window_galaxy(chi, z), sample2.window_galaxy(chi, z)
        ans += simps(w1*w2*plchi/chi**2, chi)
        if plot:
            plt.figure()
            plt.xlabel(r'$\chi$')
            plt.yscale('log')
            plt.xscale('log') if plot_xlog else None
            plt.plot(chi, w1*w2*plchi/chi**2, label='g, g')
            plt.plot(chi, w1*w2*plchi.max()/chi**2)
            print(f'Cgg(l)                                ={ans}')
        # g mag
        chi = get_chi_overlap( sample1.window_galaxy_chirange(), sample2.window_magnification_chirange(), self.chi_bin_galaxy_window)
        z = self.pk_class.get_z_from_chi(chi)
        plchi = self.pk_class.get_pkgm_lchi(l, chi, b1_1, model)
        w1, w2 = sample1.window_galaxy(chi, z), sample2.window_magnification(chi, z)
        ans += simps(w1*w2*plchi/chi**2, chi) * 2*(alpha_mag2-1)
        if plot:
            plt.plot(chi, w1*w2*plchi/chi**2 * 2*(alpha_mag2-1), label='g, mag')
            print(f'Cgg(l)+Cg,mag(l)                      ={ans}')
        # mag g
        chi = get_chi_overlap( sample1.window_magnification_chirange(), sample2.window_galaxy_chirange(), self.chi_bin_galaxy_window)
        z = self.pk_class.get_z_from_chi(chi)
        plchi = self.pk_class.get_pkgm_lchi(l, chi, b1_2, model)
        w1, w2 = sample1.window_magnification(chi, z), sample2.window_galaxy(chi, z)
        ans += simps(w1*w2*plchi/chi**2, chi) * 2*(alpha_mag1-1)
        if plot:
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
        # g E
        chi = get_chi_overlap( sample_l.window_galaxy_chirange(), sample_s.window_lensing_chirange(), self.chi_bin_galaxy_window)
        z = self.pk_class.get_z_from_chi(chi)
        plchi = self.pk_class.get_pkgm_lchi(l, chi, b1, model)
        w1, w2 = sample_l.window_galaxy(chi, z), sample_s.window_lensing(chi, z)
        ans += simps(w1*w2*plchi/chi**2, chi)
        if plot:
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
        ans += simps(w1*w2*plchi/chi**2, chi) * 2*(alpha-1)
        if plot:
            plt.plot(chi, w1*w2*plchi/chi**2 * 2*(alpha-1))
        return ans
    
    def _CEE(self, sample1, sample2, l, model='nonlin', plot=False, plot_xlog=False):
        # EE
        chi = get_chi_lensing( sample1.window_lensing_chirange(), sample2.window_lensing_chirange(), 
                              self.chi_bin_lens_kernel, l/self.k_peak/100.0)
        z = self.pk_class.get_z_from_chi(chi)
        plchi = self.pk_class.get_pkmm_lchi(l, chi, model)
        w1, w2 = sample1.window_lensing(chi, z), sample2.window_lensing(chi, z)
        ans = simps(w1*w2*plchi/chi**2, chi)
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
    
    def compute_all_Cl(self, l, model='nonlin'):
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
        for name1 in lens_samples:
            for name2 in lens_samples:
                helper(name1, name2, self.Cgg)
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
            for key in self.Cl_cache['Cl'].keys():
                fname = os.path.join(dirname, key+'.txt')
                print(f'saving {key} to {fname}')
                np.savetxt(fname, self.Cl_cache['Cl'][key])
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
        # galaxy samples
        
    def angular_correlation_function_fftlog(self, name1, name2, theta, probe, plot=False):
        l = self.Cl_cache['Cl']['l']
        Cl = self.Cl_cache['Cl'][ self.Cl_names_sep.join([name1, name2]) ]
        
        logtmin, logtmax = np.log10(theta.min())-2, np.log10(theta.max())+2
        print(logtmin, logtmax)
        fftlog = dark_emulator.pyfftlog_interface.fftlog(logrmin=logtmin, logrmax=logtmax, num=8)
        mu = self.probe_mu_dict[probe]
        input_Cl = loginterp1d(l, Cl, fftlog.k)*fftlog.k
        
        # more factors
        if probe == 'wp':
            Delta_chi_g = self.galaxy_sample_dict[name1].get_Delta_chi_g()
            input_Cl *= Delta_chi_g
        
        fftlog_out = fftlog.iHT(input_Cl, mu, 0.0, -1, 1.0/2.0/np.pi)
        #acf = loginterp1d(fftlog.r, fftlog_out, theta)
        acf = ius(fftlog.r, fftlog_out)(theta)
        
        if plot:
            plt.figure()
            plt.xlabel(r'$\theta$')
            plt.ylabel(self.probe_latex_dict[probe])
            plt.loglog(fftlog.r, fftlog_out, c='C0')
            plt.loglog(fftlog.r,-fftlog_out, c='C0',ls='--')
            plt.loglog(theta, acf, c='C1')
            plt.loglog(theta,-acf, c='C1', ls='--')
            plt.show()
        return acf
        
    def covariance_fftlog(self, names1, probe1, theta1, names2, probe2, theta2, dlnt1=None, dlnt2=None, plot=False):
        l = self.Cl_cache['Cl']['l']
        Cl1 = self.Cl_cache['Cl'][ self.Cl_names_sep.join(names1) ]
        Cl2 = self.Cl_cache['Cl'][ self.Cl_names_sep.join(names2) ]
        cl1, cl2 = np.meshgrid(Cl1*l**1.5, Cl2*l**1.5)
        input_CCl3 = cl1*cl2
        nu = 1.01
        tB = two_Bessel(l, l, input_CCl3, nu1=nu, nu2=nu, 
                        N_extrap_low=0, N_extrap_high=0, c_window_width=0.25, N_pad=0)
        mu1 = self.probe_mu_dict[probe1]
        mu2 = self.probe_mu_dict[probe2]
        if dlnt1 is None:
            dlnt1 = np.log(theta1[1]/theta1[0])
        if dlnt2 is None:
            dlnt2 = np.log(theta2[1]/theta2[0])
        print(dlnt1, dlnt2)
        t1_fine, t2_fine, cov_fine = tB.two_Bessel_binave(mu1, mu2, dlnt1, dlnt2)
        cov = interp2d(t1_fine, t2_fine, cov_fine)(theta1, theta2)
        
        if plot:
            fig = plt.figure(figsize=(10, 8))
            ax1 = fig.add_subplot(2,2,1)
            ax1.imshow(cov)
            ax1.set_title(r'${\rm Cov}$['+self.probe_latex_dict[probe1]+r'$(\theta_1)$'+
                          self.probe_latex_dict[probe2]+r'$(\theta_2)$]')

            ax2 = fig.add_subplot(2,2,2)
            ax2.loglog(theta1, np.diag(cov))
            ax2.set_xlabel(r'$\theta$')
            ax2.set_ylabel(r'${\rm Cov}$['+self.probe_latex_dict[probe1]+r'$(\theta)$'+
                           self.probe_latex_dict[probe2]+r'$(\theta)$]')
            
            ax3 = fig.add_subplot(2,2,3)
            v = np.diag(cov)
            v1, v2 = np.meshgrid(v,v)
            cc = cov/(v1*v2)**0.5 # correlation coeff
            ax3.imshow(cc)
            ax3.set_title('correlation coefficients')

            plt.tight_layout()
            plt.show()
        return cov
        
        
        
        
        
        
    
    
    
        
        
# shape noise signa^2/n or sigma^2/2/n ?