""" 
	This is a library for calculations Fisher matrices for GC.

	These is for miscellaneous functions that have nothing to do with cosmology.

	Markovic & Pourtsidou, 2016
"""

import time, datetime
from glob import glob
import numpy as np
import subprocess as sp
import scipy
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

# A function to round to significant figures, not decimal places
def roundsf(x, n):
    mask = np.logical_or(np.equal(np.abs(x),np.inf), np.equal(x,0.0))
    decades = np.power(10.0,n-np.floor(np.log10(np.abs(x)))); decades[mask] = 1.0
    return np.round(np.array(x)*decades,0)/decades

# Generate a string timestamp
def get_timestamp():
	# Calculate the number of seconds since the Brexit referrendum polls closed
	t_Brexit = '23/06/2016 21:00' # GMT/UTC, 22:00 BST
	t_Brexit = time.mktime(datetime.datetime.strptime(t_Brexit, "%d/%m/%Y %H:%M").timetuple())
	return '-'+str(int(time.time() - t_Brexit))

# A function that reads in a Fisher matrix from a file
def readfish(filename):
	header = None
	with open(filename) as f:
		filestr = f.read().strip()
	if '#' in filestr:
		# Get parameter names
		header = filestr.split('\n')[0]
		parlist = header.replace('#','').strip().strip('"').split()
		# Get fiducial values
		header = filestr.split('\n')[1]
		fidlist = [float(x) for x in header.replace('#','').strip().strip('"').split()]
	else:
		raise Exception("No header in " + filename + "!")
	data = np.genfromtxt(filename)
	return np.matrix(data), parlist, fidlist

# Function that returns a list of parameters, leaving only one list entry for every z-dependent parameter
# assumes the parameter name and redshift are separated by a '_' (an underscore)
def extract_pars(parlist, filename=False):
	Pchanged = False
	newpars = []
	for par in parlist:
		if 'lnPs' in par: 
			par = par.replace('lnPs', 'Ps')
			Pchanged = True
		par = par.split('_')[0]
		if par not in newpars: newpars.append(par)	
	if Pchanged:
		printstring = "I'm assuming you meant Ps, not lnPs"
		if filename: 
			printstring += " in "+filename+"!"
		else:
			printstring += "!"
		print(printstring)

	return newpars

# Function that returns a list of redshift bins in the Fisher matrix from a list of all the columns
# assumes the parameter name and redshift are separated by a '_' (an underscore)
def extract_zs(parlist):
	zs = []
	for par in parlist:
		pl = par.split('_')
		if len(pl)>1: 
			par = float(pl[1])
			if par not in zs: zs.append(par)	
	return zs

# Function that prints out the parameters nicely for information (not orderd as in file)
def print_pars(parlist):
	pars = extract_pars(parlist)
	str = ''
	for par in pars:
		for subpar in parlist:
			if par not in subpar: continue
			str += subpar + " "
		str+="\r\n"

	return str

# A function that prints the parameter fiducial and error
def print_err(fishmat,pars,fids,par):
	covmat = np.linalg.inv(fishmat)
	ind = pars.index(par)
	margerr = np.sqrt(covmat[ind,ind])
	fid = fids[ind]
	fracerr = margerr/fid
	return '%s = %g +/- %g (%0.2g%%)'%(par,fid,margerr,100.0*fracerr)

# Draw ellipses (from EFT Fish)
def get_ellipse(centre,CM_2D):
	# We will need coefficients for the ellipse depending on the number of sigma level we want
	CL_alpha = {
		1: 1.52, 
		2: 2.48,
		3: 3.44
	}

	# Get the errors
	FM_2D = scipy.linalg.pinv(CM_2D)
	em0 = np.sqrt(CM_2D[0,0])
	eu0 = np.sqrt(1./FM_2D[0,0])
	em1 = np.sqrt(CM_2D[1,1])
	eu1 = np.sqrt(1./FM_2D[1,1])

	# Now calculate the ellipse parameters
	a = np.sqrt(0.5*(CM_2D[0,0]+CM_2D[1,1]) + np.sqrt(0.25*(CM_2D[0,0]-CM_2D[1,1])**2 + CM_2D[0,1]**2))
	b = np.sqrt(0.5*(CM_2D[0,0]+CM_2D[1,1]) - np.sqrt(0.25*(CM_2D[0,0]-CM_2D[1,1])**2 + CM_2D[0,1]**2))
	theta = 180./np.pi * 0.5*np.arctan(2.*CM_2D[0,1] / (CM_2D[0,0]-CM_2D[1,1])) # should be in degrees

	if CM_2D[0,0]<CM_2D[1,1]:
		width=2.*b
		height=2.*a
	else:
		width=2.*a
		height=2.*b

	# Create all the ellipses
	ellipse1 = Ellipse(xy=(centre[0],centre[1]), width=width*CL_alpha[1], height=height*CL_alpha[1], angle=theta, 
					linewidth=0, fill=True, zorder=2, alpha=0.7, color='g')
	ellipse2 = Ellipse(xy=(centre[0],centre[1]), width=width*CL_alpha[2], height=height*CL_alpha[2], angle=theta, 
					linewidth=0, fill=True, zorder=2, alpha=0.2, color='g')

	return ellipse1, ellipse2, CL_alpha[1]*em0, CL_alpha[1]*em1, CL_alpha[1]*eu0, CL_alpha[1]*eu1

# Draw corner plot (from EFTFish)
# pars and fids should be lists and need to match one dimension of FishMat
def corner_plot(FishMat,pars,fids,marginalize=[],omit=[],ind=('h','ns')):	
	
	CovMat = np.linalg.inv(FishMat)
	
	fig, ax = plt.subplots(figsize=(7,7))

	p0i = pars.index(ind[0])
	c0 = fids[p0i]
	par0 = pars[p0i]
	for p1i, par1 in enumerate(pars):

		p1i = pars.index(ind[1])
		par1 = ind[1]
		c1 = fids[p1i]

		print(par0,par1)
		print(p0i,p1i)
		print(c0,c1)


		# Marginalise out everything else and make a new 2x2 Covariance Matrix
		if marginalize:
			covmat_2D = np.zeros((2,2))
			covmat_2D[0,0] = CovMat[p0i,p0i]
			covmat_2D[1,1] = CovMat[p1i,p1i]
			covmat_2D[0,1] = CovMat[p0i,p1i]
			covmat_2D[1,0] = covmat_2D[0,1]
		else:
			fishmat_2D = np.zeros((2,2))
			fishmat_2D[0,0] = FishMat[p0i,p0i]
			fishmat_2D[1,1] = FishMat[p1i,p1i]
			fishmat_2D[0,1] = FishMat[p0i,p1i]
			fishmat_2D[1,0] = fishmat_2D[0,1]
			covmat_2D = scipy.linalg.pinv(fishmat_2D)
			del fishmat_2D

		# Make a test plot
		ell1, ell2, errm0, errm1, erru0, erru1 = get_ellipse([c0,c1],covmat_2D)
		ax.axhline(c1,ls='--',c='k',lw=1); ax.axvline(c0,ls='--',c='k',lw=1) # Centre lines
		ax.axvline(c0-errm0,ls=':',c='g',lw=1); ax.axvline(c0+errm0,ls=':',c='g',lw=1) # Marginalised error lines par0
		ax.axvline(c0-erru0,ls=':',c='c',lw=1); ax.axvline(c0+erru0,ls=':',c='c',lw=1) # Marginalised error lines par0
		ax.axhline(c1-errm1,ls=':',c='g',lw=1); ax.axhline(c1+errm1,ls=':',c='g',lw=1) # Marginalised error lines par1
		ax.axhline(c1-erru1,ls=':',c='c',lw=1); ax.axhline(c1+erru1,ls=':',c='c',lw=1) # Marginalised error lines par1
		ax.add_artist(ell1)
		ax.add_artist(ell2)

		break
	
	ax.set_xlim(c0-errm0*2.1, c0+errm0*2.1)
	ax.set_ylim(c1-errm1*2.1, c1+errm1*2.1)

	plt.xlabel(par0)
	plt.ylabel(par1)

# A class that contains the Git environment at the time of it's initialisation.
# Currently it uses the subprocess module to speak to Git through the system.
#	Ideally some day it would use the GitPython module.
class GitEnv(object):

	# Optional input, /path/to/.git
	def __init__(self, git_dir=None):
		self.git_dir = git_dir
		self.hash, self.author, self.date = [str(s) for s in self.get_commit()]
		self.url = str(self.get_remote())
		self.branch = str(self.get_branch())
		self.repo = str(self.get_repo())
		self.printstart = ''
	# Also, should have an if that gives out the name of the parent folder + the
	# date and time in the case that it is NOT A GIT REPO!

	def __str__(self):
		startline = self.printstart
		as_string = startline + "This was generated by code from the Git repository at:"
		as_string += "\n" + startline + "\t " + self.url + ","
		as_string += "\n" + startline + "\t on the " + self.branch + " branch,"
		as_string += "\n" + startline + "\t with commit: " + self.hash
		as_string += "\n" + startline + "\t\t from " + self.date + ", "
		as_string += "\n" + startline + "\t\t by " + self.author + "."
		return as_string

	def set_print(self, startline):
		self.printstart = startline
	
	def get_git_cmd(self, args=[]):
		cmd = ['git']
		if self.git_dir != None:
			cmd.append('--git-dir')
			cmd.append(self.git_dir)
		for one in args:
			cmd.append(one)

		return cmd

	def get_hash(self, nochar=7, sep=''):
		return sep+self.hash[0:nochar]+sep
	
	# Get the hash, author and date of the most recent commit of the current repo.
	def get_commit(self):
		cmd = sp.Popen(self.get_git_cmd(['log', '-n','1']), stdout=sp.PIPE)
		cmd_out, cmd_err = cmd.communicate()
		newlist=[]
		for entry in cmd_out.decode("utf-8").strip().split('\n'):		
			if entry=='': continue
			entry = entry.split(' ')
			# This is a hack, should use a dict so can be sure what we are reading in:
			if 'commit' in entry[0] or 'Author' in entry[0] or 'Date' in entry[0]:
				newlist.append(' '.join(entry[1:]).strip())
		return newlist

	# At the moment this only gives the first url in what git returns.
	# Eventually it'd be nice if you could get the origin url, the fetch...
	def get_remote(self):
		cmd = sp.Popen(self.get_git_cmd(['remote', '-v']), stdout=sp.PIPE)
		cmd_out, cmd_err = cmd.communicate()
		if bool(cmd_out):
			try:
				return cmd_out.decode("utf-8").strip().split('https://')[1].split(' ')[0]
			except IndexError:
				ssh_url = cmd_out.decode("utf-8").strip().split('git@')[1].split(' ')[0]
				return ssh_url.replace(':','/')
		else:
			return 'no remote repo'

	def get_branch(self):
		cmd = sp.Popen(self.get_git_cmd(['branch']), stdout=sp.PIPE)
		cmd_out, cmd_err = cmd.communicate()
		branches = cmd_out.decode("utf-8").strip().splitlines()
		for branch in branches:
			if '*' in branch:
				return branch.replace('*','').strip()

	def get_repo(self):
		cmd = sp.Popen(self.get_git_cmd(['rev-parse','--show-toplevel']), stdout=sp.PIPE)
		cmd_out, cmd_err = cmd.communicate()
		return cmd_out.decode("utf-8").strip().split('/')[-1]
