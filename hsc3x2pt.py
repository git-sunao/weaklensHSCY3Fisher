import numpy as np
try:
    from dark_emulator_public import dark_emulator
except:
    import dark_emulator
print('using dark_emulator at ', dark_emulator.__file__)
import os, sys, json
import matplotlib.pyplot as plt
import my_python_package as mpp
from collections import OrderedDict as od
from pyhalofit import pyhalofit
from scipy.interpolate import InterpolatedUnivariateSpline as ius
from scipy.interpolate import interp2d
from time import time
from scipy.integrate import simps
from tqdm import tqdm
from scipy.special import jn
from twobessel import two_Bessel
from twobessel import log_extrap
from astropy import constants

H0 = 100 / constants.c.value *1e6 # / (Mpc/h)

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
        
    def set_cosmology_from_dict(self, cosmo_dict):
        c = np.array([[cosmo_dict[n] for n in ['omega_b', 'omega_c', 'Omega_de', 'ln10p10As', 'n_s', 'w_de']]])
        self.set_cosmology(c)
        omn = 0.00064
        h = ((c[0][0]+c[0][1]+omn)/(1-c[0][2]))**0.5
        c = {'Omega_de0': cosmo_dict['Omega_de'], 'Omega_K0': 0.0, 'w0': cosmo_dict['w_de'], 'wa': 0.0, 'h': h}
        self.halofit.set_cosmology(c)
        cosmo_dict.update({'h':h})
        self.cosmo_dict = cosmo_dict
    
    def get_cosmo_dict(self):
        return self.cosmo_dict
    
    def get_Dgrowth(self, z):
        _z = np.linspace(0.0, np.max(z), 20)
        _Dp = np.array([self.Dgrowth_from_z(__z) for __z in _z])
        return ius(_z, _Dp, ext=1)(z)
    
    def init_pklin(self, zmax, kmax=2.0):
        logk = np.linspace(-4, np.log10(kmax), 200)
        z = np.linspace(0.0, zmax, 100)
        pkL= self.get_pklin(10**logk)
        Dp = self.get_Dgrowth(z)
        self.pk_data = {'logk':logk, 'z':z, 'pkL':pkL, 'Dp':Dp}
        
    def init_pkhalo(self):
        k = 10**self.pk_data['logk']
        data = []
        for _z, _Dp in zip(self.pk_data['z'], self.pk_data['Dp']):
            pklin = self.pk_data['pkL'] * _Dp**2
            self.halofit.set_pklin(k, pklin, _z)
            pkhalo = self.halofit.get_pkhalo()
            data.append(pkhalo)
        self.pk_data['pkhalo'] = np.array(data)
        
    def get_pklin_kz(self, k, z):
        return self.get_pklin_from_z(k, z)
    
    def get_pkhalo_kz(self, k, z):
        f = interp2d(self.pk_data['logk'], self.pk_data['z'], np.log10(self.pk_data['pkhalo']), 
                     bounds_error=False, fill_value=-300)
        return 10**f(np.log10(k), z)
    
    def get_chi_from_z(self, z):
        """
        return comoving distance, chi, in Mpc/h
        """
        return self.halofit.cosmo.comoving_distance(z).value * self.cosmo_dict['h']
    
    def get_z_from_chi(self, chi):
        _z = np.linspace(0.0, 10.0, 100)
        _chi = self.get_chi_from_z(_z)
        return ius(_chi, _z)(chi)
    
    def get_pklin_lchi(self, l, chi):
        """
        return pk(k=l/chi, z=z(chi))
        """
        z = self.get_z_from_chi(chi)
        Dp = ius(self.pk_data['z'], self.pk_data['Dp'], ext=1)(z)
        pkL= 10**ius(self.pk_data['logk'], np.log10(self.pk_data['pkL']), ext=1)( np.log10(l/chi) )
        return Dp**2*pkL
    
    def get_pkhalo_lchi(self, l, chi):
        z = self.get_z_from_chi(chi)
        logk = np.log10(l/chi)
        f = interp2d(self.pk_data['logk'], self.pk_data['z'], np.log10(self.pk_data['pkhalo']), 
                     bounds_error=False, fill_value=-300)
        ans = []
        for _logk, _z in zip(logk, z):
            ans.append(10**f(_logk, _z))
        return np.array(ans)
    
    def get_pklingm_lchi(self, l, chi, b1):
        return b1*self.get_pklin_lchi(l, chi)
    
    def get_pklingg_lchi(self, l, chi, b1):
        return b1**2*self.get_pklin_lchi(l, chi)
    
    def get_pkhalogm_lchi(self, l, chi, b1):
        return b1*self.get_pkhalo_lchi(l, chi)
    
    def get_pkhalogg_lchi(self, l, chi, b1):
        return b1**2*self.get_pkhalo_lchi(l, chi)
    
    def get_pk_lchi(self, l, chi, b1, model):
        if model == 'lin':
            return self.get_pklin_lchi(l, chi, b1)
        elif model  == 'nonlin':
            return self.get_pkhalo_lchi(l, chi,  b1)
    
    def get_pkgm_lchi(self, l, chi, b1, model):
        if model == 'lin':
            return self.get_pklingm_lchi(l, chi, b1)
        elif model  == 'nonlin':
            return self.get_pkhalogm_lchi(l, chi, b1)
        
    def get_pkgg_lchi(self, l, chi, b1, model):
        if model == 'lin':
            return self.get_pklingg_lchi(l, chi, b1)
        elif model  == 'nonlin':
            return self.get_pkhalogg_lchi(l, chi,  b1)
        
#######################################
#
# galaxy sample classes
#
class galaxy_sample_source_class:
    info_keys = ['sample_name', 'z_source', 'sigma_shape', 'n2d']
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
        self.info['sample_type'] = 'source'
    
    def get_sample_info(self):
        return self.info
    
    def set_cosmology_from_dict(self, cosmo_dict):
        self.halofit.set_cosmology(cosmo_dict)
        self.Omega_m = 1.0 - cosmo_dict['Omega_de']
        z_source = self.info['z_source']
        self.chi_source = self.halofit.cosmo.comoving_distance(z_source).value * self.cosmo_dict['h']
    
    def window(self, chi, z):
        ans = np.zeros(chi.shape)
        sel = chi < self.chi_source
        ans[sel] = 3.0/2.0 * H0**2 * self.Omega_m / (1+z[sel]) * chi[sel]*(self.chi_source-chi[sel])/self.chi_source
        return ans
    
    def get_chi_range(self, r=0.01):
        return [self.chi_source*r, self.chi_source*(1-r)]

class galaxy_sample_lens_class:
    info_keys = ['sample_name', 'z_lens', 'z_min', 'z_max', 'galaxy_bias', 'n2d', 'alpha_mag']
    def __init__(self, info):
        """
        z_lens : representative comoving distance to this lens galaxy sample.
        z_min : minimum comoving distance of this galaxy sample
        z_max : maximum comoving distance of this galaxy sample
        galaxy_bias : b1
        n2d : mean nubmer density used to compute shot noise term. #/deg^2
        """
        self.set_galaxy_info(info)
        
    def set_galaxy_info(self, info):
        if isinstance(info, dict):
            self._set_galaxy_info_from_dict(info)
        elif isinstance(info, list):
            self._set_galaxy_info_from_list(info)
    
    def _set_galaxy_info_from_list(self, info_list):
        self._set_galaxy_info_from_dict(od(zip(self.info_keys, info_list)))
    
    def _set_galaxy_info_from_dict(self, info):
        self.info = info
        self.info['sample_type'] = 'lens'
    
    def get_sample_info(self):
        return self.info
    
    def set_cosmology_from_dict(self, cosmo_dict):
        self.halofit.set_cosmology(cosmo_dict)
        self.Omega_m = 1.0 - cosmo_dict['Omega_de']
        self.chi_lens = self.halofit.cosmo.comoving_distance(self.info['z_lens']).value * self.cosmo_dict['h']
        self.chi_min = self.halofit.cosmo.comoving_distance(self.info['z_min']).value * self.cosmo_dict['h']
        self.chi_max = self.halofit.cosmo.comoving_distance(self.info['z_max']).value * self.cosmo_dict['h']
    
    def window_galaxy(self, chi, z):
        ans = np.zeros(chi.shape)
        sel = np.logical_and(self.chi_min < chi, chi < self.chi_max)
        ans[sel] = 1
        return ans
    
    def window_galaxy_normalized(self, chi, z):
        return self.window_galaxy(chi, z)/(self.chi_max - self.chi_min)
    
    def window(self, chi, z):
        return self.window_galaxy_normalized(chi, z)
    
    def window_mag(self, chi, z):
        ans = np.zeros(chi.shape)
        sel = chi < self.chi_lens
        ans[sel] = 3.0/2.0 * H0**2 * self.Omega_m / (1+z[sel]) * chi[sel]*(self.chi_lens-chi[sel])/self.chi_lens
        return ans
    
    def get_chi_range(self, r=0.01):
        """
        chi range used to compute convolution with power spectrum to obtain 2d power spectrum, C(l).
        Note that even for lens galaxy sample, chi range is from 0 to chi_lens, because magnification bias effect is included.
        """
        return [self.chi_lens*r, self.chi_lens*(1-r)]

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
    def __init__(self, pk_class=None):
        self.pk_class = pk_class
        self.galaxy_sample_dict = od()
        self.cl_cache = od()
        self.cl_cache_sep = '-'
        
    def set_galaxy_sample(self, galaxy_sample):
        if galaxy_sample.info['sample_name'] in self.galaxy_sample_dict.keys():
            print(f'Note: Sample name {galaxy_sample.info["sample_name"]} is already used.\nPreviously registered sample is replaced.')
        self.galaxy_sample_dict[galaxy_sample.info['sample_name']] = galaxy_sample
        
    def get_galaxy_sample_names(self):
        return list(self.galaxy_sample_dict.keys())
        
    def set_cosmology_from_dict(self, cosmo_dict=None):
        if cosmo_dict is not None:
            self.power3d.set_cosmology_from_dict(cosmo_dict)
        cosmo_dict = self.power3d.get_cosmo_dict()
        for key in self.galaxy_sample_dict.keys():
            self.galaxy_sample_dict[key].set_cosmology_from_dict(cosmo_dict)
        self.cosmo_dict = cosmo_dict
        
    def get_zmax_from_galaxy_samples(self):
        zmax = 0
        for key, sample in self.galaxy_sample_dict.items():
            if sample.info['sample_type'] == 'source':
                _zmax = sample.info['z_source']
            elif sample.info['sample_type'] == 'lens':
                _zmax = sample.info['z_lens']
            if _zmax > zmax:
                zmax = _zmax
        return zmax
    
    def init_pk(self):
        zmax = self.get_zmax_from_galaxy_samples()
        self.pk_class.init_pklin(zmax)
        self.pk_class.init_pkhalo()
        
    def _get_chi_from_2samples(self, sample1, sample2, chi_bin=100):
        chi_min1, chi_max1 = sample1.get_chi_range()
        chi_min2, chi_max2 = sample2.get_chi_range()
        chi = np.linspace(min(chi_min1, chi_min2), max(chi_max1, chi_max2), chi_bin)
        z = self.pk_class.get_z_from_chi(chi)
        return chi, z
        
    def _Pgg2Dconvolve(self, name1, name2, l, model='nonlin', kbin=100):
        integrand = 0
        sample1, sample2 = self.galaxy_sample_dict[name1], self.galaxy_sample_dict[name2]
        b1_1, b1_2 = sample1.info['galaxy_bias'], sample2.info['galaxy_bias']
        ans = 0
        # gg
        chi, z = self._get_chi_from_2samples(sample1, sample2)
        pkgg_lz  = self.pk_class.get_pkgg_lchi(l, chi)
        integrand+= sample1.window(chi, z)*sample2.window(chi, z)*power
        ans += np.sum(integrand*k)*dlnk/l
        # gm
        chi, z = self._get_chi_from_2samples(sample1, sample2)
        pkgg_lz  = self.pk_class.get_pkgg_lchi(l, chi)
        integrand+= sample1.window(chi, z)*sample2.window(chi, z)*power
        ans += np.sum(integrand*k)*dlnk/l
        
        z, chi, k, power  = self.power3d.get_Pgm_lz(l, z_min, z_max, model=model, kbin=kbin, b1=b1_1)
        integrand+= sample1.window(chi, z)*2*(sample2.info['alpha_mag']-1)*sample2.window_mag(chi, z)*power
        # mg
        z, chi, k, power  = self.power3d.get_Pgm_lz(l, z_min, z_max, model=model, kbin=kbin, b1=b1_2)
        integrand+= 2*(sample1.info['alpha_mag']-1)*sample1.window_mag(chi, z)*sample2.window(chi, z)*power
        # mm
        z, chi, k, power  = self.power3d.get_Pmm_lz(l, z_min, z_max, model=model, kbin=kbin)
        integrand+= 2*(sample1.info['alpha_mag']-1)*2*sample1.window_mag(chi, z)*(sample2.info['alpha_mag']-1)*sample2.window_mag(chi, z)*power
        # integrate
        dlnk = np.diff(np.log(k))[0]
        ans = np.sum(integrand*k)*dlnk/l
        return ans
    
    def _PgE2Dconvolve(self, name1, name2, l, model='nonlin', kbin=100):
        integrand = 0
        sample1, sample2 = self.galaxy_sample_dict[name1], self.galaxy_sample_dict[name2]
        z_min, z_max = self.init_convolve(sample1, sample2)
        b1 = sample1.info['galaxy_bias']
        # gm
        z, chi, k, power  = self.power3d.get_Pgm_lz(l, z_min, z_max, model=model, kbin=kbin, b1=b1)
        integrand+= sample1.window(chi, z)*sample2.window(chi, z)*power
        # mm
        z, chi, k, power  = self.power3d.get_Pmm_lz(l, z_min, z_max, model=model, kbin=kbin)
        integrand+= 2*(sample1.info['alpha_mag']-1)*2*sample1.window_mag(chi, z)*sample2.window(chi, z)*power
        # integrate
        dlnk = np.diff(np.log(k))[0]
        ans = np.sum(integrand*k)*dlnk/l
        return ans
    
    def _PEE2Dconvolve(self, name1, name2, l, model='nonlin', kbin=100):
        integrand = 0
        sample1, sample2 = self.galaxy_sample_dict[name1], self.galaxy_sample_dict[name2]
        z_min, z_max = self.init_convolve(sample1, sample2)
        # mm
        z, chi, k, power  = self.power3d.get_Pmm_lz(l, z_min, z_max, model=model, kbin=kbin)
        integrand+= sample1.window(chi, z)*sample2.window(chi, z)*power
        # integrate
        dlnk = np.diff(np.log(k))[0]
        ans = np.sum(integrand*k)*dlnk/l
        return ans
    
    def _P2Dconvolve(self, func, name1, name2, l, model='nonlin', kbin=100, progress=True, desc='', cache=True):
        ans = []
        if progress:
            for _l in  tqdm(l, desc=desc):
                _ans = func(name1, name2, _l, model=model, kbin=kbin)
                ans.append(_ans)
        else:
            for _l in l:
                _ans = func(name1, name2, _l, model=model, kbin=kbin)
                ans.append(_ans)
        ans = np.reshape(np.array([ans]), l.shape)
        if cache:
            self.P2Dcache[self.P2Dcache_sep.join([name1, name2])] = {'l':l, 'P2D':ans}
        return ans
    
    def _P2Dconvolve_list(self, func_name1_name2_list, l, model='nonlin', kbin=100, progress=True, desc=''):
        """
        for fast computation for b1 model
        """
        ans_list = [[] for _ in range(len(func_name1_name2_list))]
        def compute(_l):
            for i, func_name1_name2 in enumerate(func_name1_name2_list):
                func, name1, name2 = func_name1_name2
                ans = func(name1, name2, _l, model=model, kbin=kbin)
                ans_list[i].append(ans)
        if progress:
            for _l in tqdm(l, desc=desc):
                compute(_l)
        else:
            for _l in l:
                compute(_l)
        return np.array(ans_list)
    
    def Pgg2Dconvolve(self, name1, name2, l, model='nonlin', kbin=100, progress=False, desc='Pgg', cache=True):
        return self._P2Dconvolve(self._Pgg2Dconvolve, name1, name2, l, model=model, kbin=kbin, progress=progress, desc=desc, cache=cache)
    
    def PgE2Dconvolve(self, name1, name2, l, model='nonlin', kbin=100, progress=False, desc='PgE', cache=True):
        return self._P2Dconvolve(self._PgE2Dconvolve, name1, name2, l, model=model, kbin=kbin, progress=progress, desc=desc, cache=cache)
    
    def PEE2Dconvolve(self, name1, name2, l, model='nonlin', kbin=100, progress=False, desc='PEE', cache=True):
        return self._P2Dconvolve(self._PEE2Dconvolve, name1, name2, l, model=model, kbin=kbin, progress=progress, desc=desc, cache=cache)
    
    def compute_all_P2D(self, l, model='nonlin', kbin=100, progress=False):
        names = self.get_galaxy_sample_names()
        for name1 in names:
            for name2 in names:
                if self.P2Dcache_sep.join([name1, name2]) in self.P2Dcache.keys():
                    ans = self.P2Dcache[self.P2Dcache_sep.join([name1, name2])]['P2D']
                elif self.P2Dcache_sep.join([name2, name1]) in self.P2Dcache.keys():
                    ans = self.P2Dcache[self.P2Dcache_sep.join([name2, name1])]['P2D']
                else:
                    sample1, sample2 = self.galaxy_sample_dict[name1], self.galaxy_sample_dict[name2]
                    if sample1.info['sample_type'] == 'lens' and sample2.info['sample_type'] == 'lens':
                        ans = self.Pgg2Dconvolve(name1, name2, l, model=model, kbin=kbin, progress=progress, desc=f'Pgg {name1}, {name2}', cache=False)
                    elif sample1.info['sample_type'] == 'lens' and sample2.info['sample_type'] == 'source':
                        ans = self.PgE2Dconvolve(name1, name2, l, model=model, kbin=kbin, progress=progress, desc=f'PgE {name1}, {name2}', cache=False)
                    elif sample1.info['sample_type'] == 'source' and sample2.info['sample_type'] == 'lens':
                        ans = self.PgE2Dconvolve(name2, name1, l, model=model, kbin=kbin, progress=progress, desc=f'PgE {name2}, {name1}', cache=False)
                    elif sample1.info['sample_type'] == 'source' and sample2.info['sample_type'] == 'source':
                        ans = self.PEE2Dconvolve(name1, name1, l, model=model, kbin=kbin, progress=progress, desc=f'PEE {name1}, {name2}', cache=False)
                self.P2Dcache[self.P2Dcache_sep.join([name1, name2])] = {'l':l, 'P2D':ans}
                
    def compute_all_P2D_list(self, l, model='nonlin', kbin=100, progress=True):
        names = self.get_galaxy_sample_names()
        func_name1_name2_dict = od()
        for name1 in names:
            for name2 in names:
                if not(self.P2Dcache_sep.join([name1, name2]) in func_name1_name2_dict.keys()) and not(self.P2Dcache_sep.join([name2, name1]) in func_name1_name2_dict.keys()):
                    sample1, sample2 = self.galaxy_sample_dict[name1], self.galaxy_sample_dict[name2]
                    if sample1.info['sample_type'] == 'lens' and sample2.info['sample_type'] == 'lens':
                        func_name1_name2 = [self._PgE2Dconvolve, name1, name2]
                    elif ample1.info['sample_type'] == 'lens' and sample2.info['sample_type'] == 'source':
                        func_name1_name2 = [self._PgE2Dconvolve, name1, name2]
                    elif sample1.info['sample_type'] == 'source' and sample2.info['sample_type'] == 'lens':
                        func_name1_name2 = [self._PgE2Dconvolve, name2, name1]
                    elif sample1.info['sample_type'] == 'source' and sample2.info['sample_type'] == 'source':
                        func_name1_name2 = [self._PEE2Dconvolve, name1, name2]
                func_name1_name2_dict[self.P2Dcache_sep.join([name1, name2])] = func_name1_name2
                
        func_name1_name2_list = list(func_name1_name2_dict.values())
        ans = self._P2Dconvolve_list(func_name1_name2_list, l, model=model, kbin=kbin, progress=progress, desc='')
        for i, key in enumerate(func_name1_name2_dict.keys()):
            self.P2Dcache[key] = {'l':l, 'P2D':ans[i]}
        
        for name1 in names:
            for name2 in names:
                if not self.P2Dcache_sep.join([name1, name2]) in self.P2Dcache.keys():
                    self.P2Dcache[self.P2Dcache_sep.join([name1, name2])] = self.P2Dcache[self.P2Dcache_sep.join([name2, name1])]
    
    def dump_cache(self, dirname, override=False):
        if (not os.path.exists(dirname)) or override:
            print(f'dumping result to {dirname}')
            os.makedirs(dirname) if not os.path.exists(dirname) else None
            # power spectrum class
            fname = os.path.join(dirname, 'power3d.txt')
            print(f'Saving power3d to {fname}.')
            with open(fname, 'w') as f:
                f.write(str(type(self.power3d)))
            # cosmology
            fname = os.path.join(dirname, 'cosmology.json')
            print(f'Saving cosmology to {fname}')
            json.dump(self.cosmo_dict, open(fname, 'w'), indent=3, ensure_ascii=False)
            # galaxy sample
            print(f'Saving galaxy samples to...')
            for name, sample in self.galaxy_sample_dict.items():
                fname = os.path.join(dirname, name+'.json')
                print(f'  {fname}')
                json.dump(sample.get_sample_info(), open(fname, 'w'), indent=3, ensure_ascii=False)
            # P2D(Cl)
            print(f'Saving angular power spectra to...')
            for comb, P2D in self.P2Dcache.items():
                data = np.array([P2D['l'], P2D['P2D']]).T
                fname= os.path.join(dirname, comb+'.txt')
                print(f'  {fname}')
                np.savetxt(fname, data)
        else:
            print(f'{dirname} already exists.')
            
    def load_cache(self, dirname):
        # power3d
        fname = os.path.join(dirname, 'power3d.txt')
        with open(fname, 'r') as f:
            power3d_name = f.readline()
        if power3d_name == "<class 'hsc3x2pt.power_b1_class'>":
            self.power3d = power_b1_class()  
        # galaxy sample
        sample_fnames = mpp.utility.grep(os.listdir(dirname), 'json')
        for fname in sample_fnames:
            if fname != 'cosmology.json':
                print(os.path.join(dirname, fname))
                info = json.load(open(os.path.join(dirname, fname)) , object_pairs_hook=od)
                if info['sample_type'] == 'lens':
                    sample = galaxy_sample_lens_class(info)
                elif info['sample_type'] == 'source':
                    sample = galaxy_sample_source_class(info)
                self.set_galaxy_sample(sample)
        # cosmology
        fname = os.path.join(dirname, 'cosmology.json')
        cosmo_dict = json.load(open(fname, 'r'), object_pairs_hook=od)
        self.set_cosmology_from_dict(cosmo_dict)
        # P2D
        names = self.get_galaxy_sample_names()
        for name1 in names:
            for name2 in names:
                key = self.P2Dcache_sep.join([name1, name2])
                fname = os.path.join(dirname, key+'.txt')
                print(f'loading {key}')
                d = np.loadtxt(fname)
                self.P2Dcache[key] = {'l':d[:, 0], 'P2D':d[:,1]}
                
    def getP2D(self, name1, name2, include_shot_nosise=True):
        key = self.P2Dcache_sep.join([name1, name2])
        P2D = self.P2Dcache[key]
        l = P2D['l']
        ans = P2D['P2D']
        if include_shot_nosise and name1 == name2:
            sample = self.galaxy_sample_dict[name1]
            if sample.info['sample_type'] == 'lens':
                ans += 1.0/sample.info['n2d']
            elif sample.info['sample_type'] == 'source':
                ans += sample.info['sigma_shape']**2/sample.info['n2d']/2.0 # ? 
        return l, ans
    
def J0ave(k, rmin, rmax):
    def i(kr):
        return kr*jn(1,kr)
    krmin, krmax = k*rmin, k*rmax
    norm = (krmax**2-krmin**2)/2.0
    ans = np.ones(k.shape)
    sel = k > 0
    ans[sel] = (i(krmax) - i(krmin))[sel]/norm[sel]
    return ans

def J2ave(k, rmin, rmax):
    def i(kr):
        return -2*jn(0,kr) - kr*jn(1,kr)
    krmin, krmax = k*rmin, k*rmax
    norm = (krmax**2-krmin**2)/2.0
    ans = np.zeros(k.shape)
    sel = k > 0
    ans[sel] = (i(krmax) - i(krmin))[sel]/norm[sel]
    return ans

def J4ave(k, rmin, rmax):
    def i(kr):
        return (-8/kr+kr**2)*jn(1,kr) - 8*jn(2, kr)
    krmin, krmax = k*rmin, k*rmax
    norm = (krmax**2-krmin**2)/2.0
    ans = np.zeros(k.shape)
    sel = k > 0
    ans[sel] = (i(krmax) - i(krmin))[sel]/norm[sel]
    return ans

class radial_bin_class:
    def __init__(self, theta_bins=None):
        self.set_theta_bin(theta_bins)
        
    def set_theta_bin(self, theta_bins):
        self.theta_bins = theta_bins
        
    def set_theta_bin_from_min_max_num(self, tmin, tmax, num):
        dlnt = np.log(tmax)-np.log(tmin)
        self.theta_bins = []
        for i in range(num):
            self.theta_bins.append([ np.exp(np.log(tmin)+dlnt*i/num), np.exp(np.log(tmin)+dlnt*(i+1)/num) ])
            
    def theta_representative(self):
        """
        \bar{r} = \int rdr r / \int rdr = 2.0/3.0 * (rmax-rmin) 
        """
        t = []
        for theta_bin in self.theta_bins:
            _t = 2.0/3.0*(theta_bin[1]-theta_bin[0])
            t.append(_t)
        return np.array(t)


probe_mu_dict = {'xi_plus':0, 'xi+':0, 'xi_minus':4, 'xi-':4, 'wp':0, 'gamma_t':2}
class covariance_class:
    def __init__(self, convolve2D):
        self.convolve2D = convolve2D
        
    def get_covariance(self, names1, names2, t1, t2, probe1, probe2, dlnt1=None, dlnt2=None, method='fftlog'):
        if method == 'fftlog':
            return self.get_covariance_fftlog(names1, names2, t1, t2, probe1, probe2, dlnt1=None, dlnt2=None)
        
    def get_covariance_fftlog(self, names1, names2, t1, t2, probe1, probe2, dlnt1=None, dlnt2=None):
        l1, pl1 = self.convolve2D.getP2D(names1[0], names1[1], include_shot_nosise=True)
        l2, pl2 = self.convolve2D.getP2D(names2[0], names2[1], include_shot_nosise=True)
        p1, p2 = np.meshgrid(pl1*l1**1.5, pl2*l2**1.5)
        input_pl = p1*p2
        nu = 1.01
        tB = two_Bessel(l1, l2, input_pl, nu1=nu, nu2=nu, N_extrap_low=0, N_extrap_high=0, c_window_width=0.25, N_pad=0)
        mu1 = probe_mu_dict[probe1]
        mu2 = probe_mu_dict[probe2]
        if dlnt1 is None:
            dlnt1 = np.log(t1[1]/t1[0])
        if dlnt2 is None:
            dlnt2 = np.log(t2[1]/t2[0])
        t1_fine, t2_fine, cov_fine = tB.two_Bessel_binave(mu1, mu2, dlnt1, dlnt2)
        cov = interp2d(t1_fine, t2_fine, cov_fine)(t1, t2)
        return cov
        
class signal_class:
    def __init__(self, convolve2D):
        self.convolve2D = convolve2D
    
    def get_signal_fftlog(self, name1, name2, t, probe):
        tmin, tmax = np.min(t) / 1e3, np.max(t) * 1e3
        lmin, lmax = 1/tmax, 1/tmin
        print(np.log10(tmin), np.log10(tmax))
        fftlog = dark_emulator.pyfftlog_interface.fftlog(logrmin=np.log10(tmin), logrmax=np.log10(tmax), num=8)
        
        l, pl = self.convolve2D.getP2D(name1, name2, include_shot_nosise=False)
        dlnl = np.log(l[1]/l[0])
        N_small, N_large = np.log(l[0]/lmin)/dlnl*1.1, np.log(lmax/l[-1])/dlnl*1.1
        l_extrap = log_extrap(l, N_small, N_large)
        pl_extrap= log_extrap(pl,N_small, N_large)
        input_pk = fftlog.k*ius(l_extrap, pl_extrap, ext=1)(fftlog.k)
        mu = probe_mu_dict[probe]
        ans = ius(fftlog.r, fftlog.iHT(input_pk, mu, 0.0, -1, 1.0/2.0/np.pi))(t)
        return ans
        
        
# shape noise signa^2/n or sigma^2/2/n ?