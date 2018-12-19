******************************************************************************************

# HTRU2

Author: Rob Lyon, School of Computer Science & Jodrell Bank Centre for Astrophysics,
		University of Manchester, Kilburn Building, Oxford Road, Manchester M13 9PL.

Contact:	rob@scienceguyrob.com or robert.lyon@.manchester.ac.uk
Web:		http://www.scienceguyrob.com or http://www.cs.manchester.ac.uk
			or alternatively http://www.jb.man.ac.uk
******************************************************************************************

1. Overview

	HTRU2 is a data set which describes a sample of pulsar candidates collected during the
	High Time Resolution Universe Survey (South) [1]. 
	
	Pulsars are a rare type of Neutron star that produce radio emission detectable here on
	Earth. They are of considerable scientific interest as probes of space-time, the inter-
	stellar medium, and states of matter (see [2] for more uses). 
	
	As pulsars rotate, their emission beam sweeps across the sky, and when this crosses
	our line of sight, produces a detectable pattern of broadband radio emission. As pulsars
	rotate rapidly, this pattern repeats periodically. Thus pulsar search involves looking
	for periodic radio signals with large radio telescopes.
	
	Each pulsar produces a slightly different emission pattern, which varies slightly with each
	rotation (see [2] for an introduction to pulsar astrophysics to find out why). Thus a 
	potential signal detection known as a 'candidate', is averaged over many rotations of the
	pulsar, as determined by the length of an observation. In the absence of additional info,
	each candidate could potentially describe a real pulsar. However in practice almost all
	detections are caused by radio frequency interference (RFI) and noise, making legitimate
	signals hard to find.
	
	Machine learning tools are now being used to automatically label pulsar candidates to
	facilitate rapid analysis. Classification systems in particular are being widely adopted,
	(see [4,5,6,7,8,9]) which treat the candidate data sets  as binary classification problems.
	Here the legitimate pulsar examples are a minority positive class, and spurious examples
	the majority negative class. At present multi-class labels are unavailable, given the
	costs associated with data annotation.
	
	The data set shared here contains 16,259 spurious examples caused by RFI/noise, and 1,639
	real pulsar examples. These examples have all been checked by human annotators. Each
	candidate is described by 8 continuous variables. The first four are simple statistics
	obtained from the integrated pulse profile (folded profile). This is an array of continuous
	variables that describe a longitude-resolved version of the signal that has been averaged
	in both time and frequency (see [3] for more details). The remaining four variables are
	similarly obtained from the DM-SNR curve (again see [3] for more details). These are 
	summarised below:
	
	1. Mean of the integrated profile.
	2. Standard deviation of the integrated profile.
	3. Excess kurtosis of the integrated profile.
	4. Skewness of the integrated profile.
	5. Mean of the DM-SNR curve.
	6. Standard deviation of the DM-SNR curve.
	7. Excess kurtosis of the DM-SNR curve.
	8. Skewness of the DM-SNR curve.
	
	HTRU 2 Summary
	
	17,898 total examples.
	1,639 positive examples.
	16,259 negative examples.
	
	
	The data is presented in two formats: CSV and ARFF (used by the WEKA data mining tool).
	Candidates are stored in both files in separate rows. Each row lists the variables first,
	and the class label is the final entry. The class labels used are 0 (negative) and 1 
	(positive).
	
	Please not that the data contains no positional information or other astronomical details. It is 
	simply feature data extracted from candidate files using the PulsarFeatureLab tool (see [10]).

2. Citing our work	
	
	If you use the dataset in your work please cite us using the DOI of the dataset, and the paper:
	
	R. J. Lyon, B. W. Stappers, S. Cooper, J. M. Brooke, J. D. Knowles, Fifty Years of Pulsar
	Candidate Selection: From simple filters to a new principled real-time classification approach
	MNRAS, 2016.
	
3. Acknowledgements

	This data was obtained with the support of grant EP/I028099/1 for the University of Manchester 
	Centre for Doctoral Training in Computer Science, from the UK Engineering and Physical Sciences
	Research Council (EPSRC). The raw observational data was collected by the High Time Resolution
	Universe Collaboration using the Parkes Observatory, funded by the Commonwealth of Australia and
	managed by the CSIRO.
	
4. References

	[1] M.~J. Keith et al., "The High Time Resolution Universe Pulsar Survey - I. System Configuration 
	    and Initial Discoveries",2010, Monthly Notices of the Royal Astronomical Society, vol. 409,
	    pp. 619-627. DOI: 10.1111/j.1365-2966.2010.17325.x
	
	[2] D. R. Lorimer and M. Kramer, "Handbook of Pulsar Astronomy", Cambridge University Press, 2005.
	
	[3] R. J. Lyon, "Why Are Pulsars Hard To Find?", PhD Thesis, University of Manchester, 2015.
	
	[4] R. J. Lyon et al., "Fifty Years of Pulsar Candidate Selection: From simple filters to a new
		principled real-time classification approach", Monthly Notices of the Royal Astronomical Society,
		submitted.
		
	[5] R. P. Eatough et al., "Selection of radio pulsar candidates using artificial neural networks",
		Monthly Notices of the Royal Astronomical Society, vol. 407, no. 4, pp. 2443-2450, 2010.
		
	[6] S. D. Bates et al., "The high time resolution universe pulsar survey vi. an artificial neural
		network and timing of 75 pulsars", Monthly Notices of the Royal Astronomical Society, vol. 427,
		no. 2, pp. 1052-1065, 2012.

	[7] D. Thornton, "The High Time Resolution Radio Sky", PhD thesis, University of Manchester,
		Jodrell Bank Centre for Astrophysics School of Physics and Astronomy, 2013.
		
	[8] K. J. Lee et al., "PEACE: pulsar evaluation algorithm for candidate extraction a software package
		for post-analysis processing of pulsar survey candidates", Monthly Notices of the Royal Astronomical
		Society, vol. 433, no. 1, pp. 688-694, 2013.
		
	[9] V. Morello et al., "SPINN: a straightforward machine learning solution to the pulsar candidate
		selection problem", Monthly Notices of the Royal Astronomical Society, vol. 443, no. 2,
		pp. 1651-1662, 2014.
		
	[10] R. J. Lyon, "PulsarFeatureLab", 2015, https://dx.doi.org/10.6084/m9.figshare.1536472.v1.
