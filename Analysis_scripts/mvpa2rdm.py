# -*- coding: utf-8 -*-
"""
Module for reading in MVPA matrix of single trial fMRI data
(trials x voxels) and transforming it to a Representational
Dissimilarity Matrix (RDM).

Part of the Research Master Psychology course 
'Programming: The Next Step' at the University of Amsterdam.

Lukas Snoek, spring 2015
"""

import os
import cpickle
import numpy as np
import glob
import glm2mvpa




