import argparse
import importlib
import os
import math
import sys
import random
import json

from LDMX.Framework import EventTree
from LDMX.Framework import ldmxcfg

from ROOT import TCanvas, TH3F, TH1F, TEfficiency, TGraph2D, TTree, TFile

#import matplotlib as plt
import xgboost as xgb
import pickle as pkl
import numpy as np

#plt.use('Agg')
from optparse import OptionParser
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

##############################################################################

class sampleContainer:
    def __init__(self, dn, EDPath, isBkg, bdt):
        print("Initializing Container!")
        self.isBkg = isBkg
        self.events = []
        evtcount = 0

        rootfile = TFile(EDPath + "event_histograms.root", "recreate")

        h0 = TH1F("totalevents", "totalevents", 40, 0, 2000)
        h1 = TH1F("passtrigger", "passtrigger", 40, 0, 2000)
        h2 = TH1F("hcalEnergyReq", "hcalEnergyReq", 40, 0, 2000)
        h3 = TH1F("passTrackerVeto", "passTrackerVeto", 40, 0, 2000)
        h4 = TH1F("passBDT", "passBDT", 40, 0, 2000)
        h5 = TH1F("EvetoDisc","EvetoDisc",100,0,1) #clocks the Eveto Disc for events passing Hcal BDT
        h6 = TH1F("passEveto","passEveto",40, 0, 2000) #events passing the Ecal Veto cut
        h7 = TH1F("Nstraight","Nstraight",10,-1,9) #straight tracks
        h8 = TH1F("NLinreg","NLinreg",10,-1,9) #regular linear tracks 
        

        for filename in os.listdir(dn):
            fn = os.path.join(dn, filename)
            tree = EventTree.EventTree(fn)
            for event in tree:

                if isBkg:
                    EcalRecHits = event.EcalRecHits_sim
                    HcalRecHits = event.HcalRecHits_sim
                    SimParticles = event.SimParticles_sim
                    TargetScoringPlaneHits = event.TargetScoringPlaneHits_sim
                    #RecoilSimHits = event.RecoilSimHits_sim
                    EcalVeto = event.EcalVeto_sim

                else:
                    EcalRecHits = event.EcalRecHits_v14
                    HcalRecHits = event.HcalRecHits_v14
                    SimParticles = event.SimParticles_v14
                    TargetScoringPlaneHits = event.TargetScoringPlaneHits_v14
                    RecoilSimHits = event.RecoilSimHits_v14
                    EcalVeto = event.EcalVeto_v14
                    
 
                evt = []

                Eupstream = 0
                Edownstream = 0
                EHcal = 0

                decayz = 1
                for it in SimParticles:
                    parents = it.second.getParents()
                    for track_id in parents:
                        if track_id == 0 and it.second.getPdgID() == -11:
                            decayz = it.second.getVertex()[2]

                for hit in HcalRecHits:
                    if hit.getZPos() >= 870:
                        EHcal += 12*hit.getEnergy()
                        
                for hit in EcalRecHits:
                    if hit.getZPos() > 500:
                        Edownstream += hit.getEnergy()
                    else:
                        Eupstream += hit.getEnergy()

                h0.Fill(decayz)
                if Eupstream < 3160: #visible trigger
                    h1.Fill(decayz)
                    #print("Event passing trigger\n")


                if Eupstream < 3160 and EHcal > 4840: #Hcal energy requirement
                    h2.Fill(decayz)
                    recoil_E = 0
                    e_list = []
                    for sphit in TargetScoringPlaneHits:
                        if sphit.getPosition()[2] > 0:
                            for it in SimParticles:
                                if it.first == sphit.getTrackID():
                                    if it.second.getPdgID() == 11:
                                        e_list.append(sphit.getEnergy())

                    if len(e_list) > 0:
                        recoil_E = max(e_list)

                    if recoil_E < 2400: #recoil tracker veto
                        h3.Fill(decayz)
                    
                        layershit = []
    
                        hits = 0
                        isohits = 0
                        isoE = 0
                        
                        xmean = 0
                        ymean = 0
                        zmean = 0
                        rmean = 0
                    
                        xmean_equal = 0
                        ymean_equal = 0
                        zmean_equal = 0
                        rmean_equal = 0

                        rms_r = 0
                    
                        xstd = 0
                        ystd = 0
                        zstd = 0

                        xstd_equal = 0
                        ystd_equal = 0
                        zstd_equal = 0
                        
                        Etot = 0
                    
                        for it in SimParticles:
                            for sphit in TargetScoringPlaneHits:
                                if sphit.getPosition()[2] > 0:
                                    if it.first == sphit.getTrackID():
                                        if isBkg:
                                            if sphit.getPdgID() == 11 and 0 in it.second.getParents():
                                                x0_gamma = sphit.getPosition()
                                                p_gamma = [-sphit.getMomentum()[0], -sphit.getMomentum()[1], 4000 - sphit.getMomentum()[2]]
                                        else:
                                            if sphit.getPdgID() == 622:
                                                x0_gamma = sphit.getPosition()
                                                p_gamma = sphit.getMomentum()
                        downstreamrmean_gammaproj = 0
                        
                        
                        for hit in HcalRecHits:
                            if hit.getZPos() >= 870:
                                hits += 1
                                x = hit.getXPos()
                                y = hit.getYPos()
                                z = hit.getZPos()
                                r = math.sqrt(x*x + y*y)
                        
                                energy = hit.getEnergy()
                                Etot += energy
                                
                                xmean += x*energy
                                ymean += y*energy
                                zmean += z*energy

                                xmean_equal += x
                                ymean_equal += y
                                zmean_equal += z

                                rms_r += r*r
                        
                                if not z in layershit:
                                    layershit.append(z)
                                
                                x_proj = x0_gamma[0] + (z - x0_gamma[2])*p_gamma[0]/p_gamma[2]
                                y_proj = x0_gamma[1] + (z - x0_gamma[2])*p_gamma[1]/p_gamma[2]
                                projdist = math.sqrt((x-x_proj)**2 + (y-y_proj)**2)
                                downstreamrmean_gammaproj += projdist*energy
                        
                                closestpoint = 9999
                                for hit2 in HcalRecHits:
                                    if abs(z - hit2.getZPos()) < 1:
                                        sepx = math.sqrt((x-hit2.getXPos())**2)
                                        sepy = math.sqrt((y-hit2.getYPos())**2)
                                        if sepx > 0 and sepx%50 == 0:
                                            if sepx < closestpoint:
                                                closestpoint = sepx
                                        elif sepy > 0 and sepy%50 == 0:
                                            if sepy < closestpoint:
                                                closestpoint = sepy
                                if closestpoint > 50:
                                    isohits += 1
                                    isoE += energy
                            
                        xmean /= Etot
                        ymean /= Etot
                        zmean /= Etot
                        rmean /= Etot

                        xmean_equal /= hits
                        ymean_equal /= hits
                        zmean_equal /= hits
                        rmean_equal /= hits
                        rms_r /= hits
                        
                        downstreamrmean_gammaproj /= Etot    
                     
                        for hit in HcalRecHits:
                            if hit.getZPos() >= 870:
                                x = hit.getXPos()
                                y = hit.getYPos()
                                z = hit.getZPos()
                                energy = hit.getEnergy()

                                xstd += energy*(x-xmean)**2
                                ystd += energy*(y-ymean)**2
                                zstd += energy*(z-zmean)**2

                                xstd_equal += (x-xmean_equal)**2
                                ystd_equal += (y-ymean_equal)**2
                                zstd_equal += (z-zmean_equal)**2

                        xstd = math.sqrt(xstd/Etot)
                        ystd = math.sqrt(ystd/Etot)
                        zstd = math.sqrt(zstd/Etot)

                        xstd_equal = math.sqrt(xstd_equal/hits)
                        ystd_equal = math.sqrt(ystd_equal/hits)
                        zstd_equal = math.sqrt(zstd_equal/hits)

                        rms_r = math.sqrt(rms_r)
                    

                        #evt.append(rms_r) 
                        evt.append(len(layershit)) #0
                  
                        evt.append(xstd) #1
                        evt.append(ystd) #2
                        evt.append(zstd) #3
                               
                        evt.append(xmean) #4
                        evt.append(ymean) #5
                        evt.append(rmean) #6

                        #evt.append(zstd_equal) 
                        #evt.append(xstd_equal) 
                        #evt.append(ystd_equal) 

                        #evt.append(rmean_equal) 
                        #evt.append(xmean_equal) 
                        #evt.append(ymean_equal) 
                        
                        evt.append(isohits) #7
                        evt.append(isoE) #8
                        evt.append(hits) #9
                        
                        evt.append(Etot) #10
                    
                        evt.append(downstreamrmean_gammaproj) #11

                        if bdt.predict(xgb.DMatrix(np.vstack((evt,np.zeros_like(evt))),np.zeros(2)))[0] >= 0.9997875:
                            h4.Fill(decayz)
                            h5.Fill(EcalVeto.getDisc())
                            h7.Fill(EcalVeto.getNStraightTracks())
                            h8.Fill(EcalVeto.getNLinRegTracks())
                            evtcount += 1 #event count is okay!
                                
                            #only grab collections if there is a high disc value
                            if (EcalVeto.getDisc() > 0.8):
                                h6.Fill(decayz)
                                ecalrechits = []
                                for hit in EcalRecHits:
                                    ecalrechits.append([[hit.getXPos(), hit.getYPos(), hit.getZPos()], hit.getEnergy()])

                                hcalrechits = []
                                for hit in HcalRecHits:
                                    hcalrechits.append([[hit.getXPos(), hit.getYPos(), hit.getZPos()], hit.getEnergy()])
                            
                                simparticles = []
                                for it in SimParticles:
                                    daughters = []
                                    for daughter in it.second.getDaughters():
                                        daughters.append(daughter)
                                    parents = []
                                    for parent in it.second.getParents():
                                        parents.append(parent)
                                
                                    simparticles.append([it.first,
                                                 it.second.getEnergy(),
                                                 it.second.getPdgID(),
                                                 [it.second.getVertex()[0], it.second.getVertex()[1], it.second.getVertex()[2]],
                                                 [it.second.getEndPoint()[0], it.second.getEndPoint()[1], it.second.getEndPoint()[2]],
                                                 [it.second.getMomentum()[0], it.second.getMomentum()[1], it.second.getMomentum()[2]],
                                                 it.second.getMass(),
                                                 it.second.getCharge(),
                                                 daughters,
                                                 parents])

                                targetscoringplanehits = []
                                for sphit in TargetScoringPlaneHits:
                                    targetscoringplanehits.append([[sphit.getPosition()[0], sphit.getPosition()[1], sphit.getPosition()[2]],
                                                           sphit.getEnergy(),
                                                           [sphit.getMomentum()[0], sphit.getMomentum()[1], sphit.getMomentum()[2]],
                                                           sphit.getTrackID(),
                                                           sphit.getPdgID()])
                            
                                ecalveto = []
                                ecalveto.append([EcalVeto.getDisc(),
                                        EcalVeto.getNStraightTracks(),
                                        EcalVeto.getNLinRegTracks()])


                                eventinfo = {'EcalRecHits': ecalrechits,
                                     'HcalRecHits': hcalrechits,
                                     'SimParticles': simparticles,
                                     'TargetScoringPlaneHits': targetscoringplanehits,
                                     'EcalVeto': ecalveto}
                            
                            
                                with open(EDPath + 'eventinfo_{0}.txt'.format(evtcount), 'w') as convert_file:
                                    convert_file.write(json.dumps(eventinfo))
                            
                                print("Wrote event {0} to file".format(evtcount))


        rootfile.cd()
        h0.Write()
        h1.Write()
        h2.Write()
        h3.Write()
        h4.Write()
        h5.Write()
        h6.Write()
        h7.Write()
        h8.Write()
        rootfile.Write()
        rootfile.Close()
        
        del h0
        del h1
        del h2
        del h3
        del h4
        del h5
        del h6
        del h7
        del h8



if __name__ == '__main__':

    parser = OptionParser()

    parser.add_option('--max_evt', dest='max_evt', type='int', default=100, help='Max Events to load')
    parser.add_option('--bdt_path', dest='bdt_path', default='/sfs/qumulo/qhome/wer2ct/LDMX/ldmx-analysis/bdt_8gev/weights/bdt_v2_0/bdt_v2_0_weights.pkl', help='BDT model to use')
    parser.add_option('--evtdisplay_path', dest='evtdisplay_path', default='/scratch/wer2ct/EcalPN_bdt/passing/', help='Where to put events that pass veto')
    
    parser.add_option('--bkg_dir', dest='bkg_dir', default='/project/hep_aag/ldmx/background/8GeV/v3.3.3_ecalPN_tskim-batch2/', help='name of background file directory')
    parser.add_option('--sig_dir', dest='sig_dir', default='/project/hep_aag/ldmx/ap/visibles/produced/8_GeV/v14/m100_testing/', help='name of signal file directory')


    (options, args) = parser.parse_args()

    # load bdt model from pkl file
    gbm = pkl.load(open(options.bdt_path, 'rb'))

    print('Loading bkg_file = ', options.bkg_dir)
    #print('Loading sig_file = ', options.sig_dir)
    bkgContainer = sampleContainer(options.bkg_dir, options.evtdisplay_path, True, gbm)
    #sigContainer = sampleContainer(options.sig_dir, options.evtdisplay_path, False, gbm)
