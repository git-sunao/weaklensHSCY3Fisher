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
        
    def _get_Pmm_lin(self, k, z):
        return self.get_pklin_from_z(k, z)

    def _get_Pmm_nonlin(self, k, z):
        _k = np.logspace(min([-4, np.log10(np.min(k))]), max([np.log10(np.max(k)*1.01), 1]), 100)
        _pklin = self._get_Pmm_lin(_k, z)
        self.halofit.set_pklin(_k, _pklin, z)
        _pkhalo = self.halofit.get_pkhalo()
        ans = ius(_k, _pkhalo, ext=1)(k)
        return ans
        
    def get_Pmm(self, k, z, model='nonlin'):
        if model == 'lin':
            ans = self._get_Pmm_lin(k,z)
        elif model == 'nonlin':
            ans = self._get_Pmm_nonlin(k,z) 
        else:
            ans = np.zeros(k.shape)
        return ans

    def _get_Pmm_lz(self, l, z_min, z_max, kbin=100, model='nonlin'):
        chi_min, chi_max = self.get_chi(z_min), self.get_chi(z_max)
        k_max, k_min = l/chi_min, l/chi_max
        k = np.logspace(np.log10(k_min), np.log10(k_max), kbin)
        chi = l/k
        z_temp = np.linspace(z_min/1.1, z_max*1.1, 100)
        chi_temp = self.get_chi(z_temp)
        args = np.argsort(chi_temp)
        z = ius(chi_temp, z_temp)(chi)
        if model == 'lin':
            sel = np.argsort(k)
            ans = np.zeros(k.shape)
            pkL = self.get_pklin(k[sel])
            Dp = np.array([self.Dgrowth_from_z(_z) for _z in z[sel]])
            ans[sel] = pkL*Dp**2
        elif model == 'nonlin':
            samp = np.array([self.get_Pmm(k, z[i], model='nonlin')[i] for i in range(len(z))])
            args = np.argsort(chi)
            ans = ius(chi[args], samp[args])(chi)
        return z, chi, k, ans

    def get_Pmm_lz(self, l, z_min, z_max, kbin=100, model='nonlin'):
        """
        Compute
        P(l/chi, z(chi))
        """
        def update(l, z_min, z_max, kbin, model):
            z, chi, k, ans = self._get_Pmm_lz(l, z_min, z_max, kbin=kbin, model=model)
            self.current_Pmm_lz_setup = od([('l', l), ('z_min', z_min), ('z_max', z_max), ('kbin', kbin) , ('model', model)])
            self.current_Pmm_lz = [z, chi, k, ans]
            return z, chi, k, ans
        if hasattr(self, 'current_Pmm_lz_setup'):
            if list(self.current_Pmm_lz_setup.values()) == [l,z_min,z_max, kbin, model]:
                #print('same setup of lz.')
                return self.current_Pmm_lz
            else:
                return update(l, z_min, z_max, kbin, model)
        else:
            return update(l, z_min, z_max, kbin, model)
        
    def get_Pgm(self, k, z, b1, model='nonlin'):
        return b1*self.get_Pmm(k, z, model=model)
    
    def get_Pgm_lz(self, l, z_min, z_max, b1, kbin=100, model='nonlin'):
        z, chi, k, ans = self.get_Pmm_lz(l, z_min, z_max, kbin=kbin, model=model)
        return z, chi, k, ans*b1
    
    def get_Pgg_lz(self, l, z_min, z_max, b1_1, b1_2, kbin=100, model='nonlin'):
        z, chi, k, ans = self.get_Pmm_lz(l, z_min, z_max, kbin=kbin, model=model)
        return z, chi, k, ans*b1_1*b1_2
    
    def get_chi(self, z):
        """
        return comoving distance, chi, in Mpc/h
        """
        return self.halofit.cosmo.comoving_distance(z).value * self.cosmo_dict['h']
    
class galaxy_sample_source_class:
    info_keys = ['sample_name', 'chi_source', 'sigma_shape', 'n2d']
    def __init__(self, info):
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
        self.info['sample_type'] = 'source'
    
    def get_sample_info(self):
        return self.info
    
    def set_cosmology_from_dict(self, cosmo_dict):
        self.Omega_m = 1.0 - cosmo_dict['Omega_de']
    
    def window(self, chi, z):
        ans = np.zeros(chi.shape)
        sel = chi < self.info['chi_source']
        ans[sel] = 3.0/2.0 * H0**2 * self.Omega_m / (1+z[sel]) * chi[sel]*(self.info['chi_source']-chi[sel])/self.info['chi_source']
        return ans
    
    def get_chi_range(self, r=0.01):
        return [self.info['chi_source']*r, self.info['chi_source']*(1-r)]

class galaxy_sample_lens_class:
    info_keys = ['sample_name', 'chi_lens', 'chi_g_min', 'chi_g_max', 'galaxy_bias', 'n2d', 'alpha_mag']
    def __init__(self, info):
        """
        chi_lens : representative comoving distance to this lens galaxy sample.
        chi_g_min : minimum comoving distance of this galaxy sample
        chi_g_max : maximum comoving distance of this galaxy sample
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
        self.Omega_m = 1.0 - cosmo_dict['Omega_de']
    
    def window_galaxy(self, chi, z):
        ans = np.zeros(chi.shape)
        sel = np.logical_and(self.info['chi_g_min'] < chi, chi < self.info['chi_g_max'])
        ans[sel] = 1
        return ans
    
    def window_galaxy_normalized(self, chi, z):
        return self.window_galaxy(chi, z)/(self.info['chi_g_max'] - self.info['chi_g_min'])
    
    def window(self, chi, z):
        return self.window_galaxy_normalized(chi, z)
    
    def window_mag(self, chi, z):
        ans = np.zeros(chi.shape)
        sel = chi < self.info['chi_lens']
        ans[sel] = 3.0/2.0 * H0**2 * self.Omega_m / (1+z[sel]) * chi[sel]*(self.info['chi_lens']-chi[sel])/self.info['chi_lens']
        return ans
    
    def get_chi_range(self, r=0.01):
        """
        chi range used to compute convolution with power spectrum to obtain 2d power spectrum, C(l).
        Note that even for lens galaxy sample, chi range is from 0 to chi_lens, because magnification bias effect is included.
        """
        return [self.info['chi_lens']*r, self.info['chi_lens']*(1-r)]

class convolve2D_class:
    """
    This class compute a convolution like
    ::math::
        C(l) = P(l) = \int{\rm d}\chi W_1(\chi )W_2(\chi)P\left(\frac{l}{\chi}, z(\chi)\right)
    """
    def __init__(self, power3d=None):
        self.power3d = power3d
        self.galaxy_sample_dict = od()
        self.P2Dcache = od()
        self.P2Dcache_sep = '-'
        
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
        
    def init_convolve(self, sample1, sample2):
        chi_r1, chi_r2 = sample1.get_chi_range(), sample2.get_chi_range()
        #chi_range = [min([chi_r1[0], chi_r2[0]]), max([chi_r1[1], chi_r2[1]])]
        chi_range = [max([chi_r1[0], chi_r2[0]])/1.1, min([chi_r1[1], chi_r2[1]])*1.1]
        z = np.linspace(0.0, 5.0, 100)
        chi = self.power3d.get_chi(z)
        args = np.argsort(chi)
        chi_of_z = ius(chi, z)
        z_range = chi_of_z(chi_range)
        z_min, z_max = np.min(z_range), np.max(z_range)
        return z_min, z_max
    
    def _Pgg2Dconvolve(self, name1, name2, l, model='nonlin', kbin=100):
        integrand = 0
        sample1, sample2 = self.galaxy_sample_dict[name1], self.galaxy_sample_dict[name2]
        z_min, z_max = self.init_convolve(sample1, sample2)
        b1_1, b1_2 = sample1.info['galaxy_bias'], sample2.info['galaxy_bias']
        # gg
        z, chi, k, power  = self.power3d.get_Pgg_lz(l, z_min, z_max, model=model, kbin=kbin, b1_1=b1_1, b1_2=b1_2)
        integrand+= sample1.window(chi, z)*sample2.window(chi, z)*power
        # gm
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