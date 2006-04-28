/***************************************************************************
 *
 * Authors:    Sjors Scheres           scheres@cnb.uam.es (2004)
 *
 * Unidad de Bioinformatica del Centro Nacional de Biotecnologia , CSIC
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or   
 * (at your option) any later version.                                 
 *                                                                     
 * This program is distributed in the hope that it will be useful,     
 * but WITHOUT ANY WARRANTY; without even the implied warranty of      
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the       
 * GNU General Public License for more details.                        
 *                                                                     
 * You should have received a copy of the GNU General Public License   
 * along with this program; if not, write to the Free Software         
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA            
 * 02111-1307  USA                                                     
 *                                                                     
 *  All comments concerning this program package may be sent to the    
 *  e-mail address 'xmipp@cnb.uam.es'                                  
 ***************************************************************************/
#include "../Prog_Refine3d.hh"


// Read ===================================================================
void Prog_Refine3d_prm::read(int argc, char **argv)  {

  if (check_param(argc,argv,"-show_all_ML_options")) {
    Prog_MLalign2D_prm ML_prm;
    ML_prm.extended_usage(true);
  }
  if (check_param(argc,argv,"-show_all_ART_options")) {
    Basic_ART_Parameters   art_prm;
    art_prm.usage_more();
    exit(0);
  }

  //Read Refine3d parameters
  fn_sel=get_param(argc,argv,"-i");
  SF.read(fn_sel);
  fn_root=get_param(argc,argv,"-o","MLrefine3D");
  fn_vol=get_param(argc,argv,"-vol");
  // Fill volume selfile
  if (Is_VolumeXmipp(fn_vol)) {
    SFvol.reserve(1);
    SFvol.insert(fn_vol);
  } else {
    SFvol.read(fn_vol);
  }
  Nvols=SFvol.ImgNo();

  angular=AtoF(get_param(argc,argv,"-ang","10"));
  fn_sym=get_param(argc,argv,"-sym","");
  lowpass=AtoF(get_param(argc,argv,"-filter","-1"));
  mask_radius=AtoF(get_param(argc,argv,"-mask_radius","-1"));
  eps=AtoF(get_param(argc,argv,"-eps","5e-5"));
  verb=AtoI(get_param(argc,argv,"-verb","1"));
  Niter=AtoI(get_param(argc,argv,"-iter","100"));

  // Hidden for now
  istart=AtoI(get_param(argc,argv,"-istart","1"));
  fn_solv=get_param(argc,argv,"-solvent","");
  do_wbp=check_param(argc,argv,"-WBP");
  tilt_range0=AtoF(get_param(argc,argv,"-tilt0","0."));
  tilt_rangeF=AtoF(get_param(argc,argv,"-tiltF","90."));

  // Checks
  if (lowpass>0.5) 
    REPORT_ERROR(1,"Digital frequency for filter should be between 0 and 0.5");

}
// Usage ===================================================================
void Prog_Refine3d_prm::usage() {
  cerr << "Usage:  Refine3d [options] "<<endl;
  cerr << "   -i <selfile>                : Selfile with input images \n"
       << "   -vol <volume/selfile>       : Initial reference volume \n"
       << "                               :  OR selfile with multiple reference volumes\n"
       << " [ -o <root=\"MLrefine3D\"> ]    : Output rootname \n"
       << " [ -ang <float=10> ]           : Angular sampling (degrees) \n"
       << " [ -iter <int=100> ]           : Maximum number of iterations \n"
       << " [ -l <float=0.2> ]            : wlsART-relaxation parameter (lambda)  \n"
       << " [ -k <float=0.5> ]            : wlsART-relaxation parameter for residual (kappa)\n"
       << " [ -n <int=10> ]               : Number of wlsART-iterations \n"
       << " [ -sym <symfile> ]            : Enforce symmetry \n"
       << " [ -mask_radius <rad.=-1> ]    : Radius for masking volume \n"
       << " [ -filter <dig.freq.=-1> ]    : Low-pass filter volume every iteration \n"
       << " [ -show_all_ML_options ]      : Show all parameters for the ML-refinement\n"
       << " [ -show_all_ART_options ]     : Show all parameters for the wlsART reconstruction \n";

}
// Show ======================================================================
void Prog_Refine3d_prm::show() {

  if (verb>0) {
    // To screen
    cerr << " -----------------------------------------------------------------"<<endl;
    cerr << " | Read more about this program in the following publication:    |"<<endl;
    cerr << " |  Scheres ea. (2006) in preparation ...                        |"<<endl;
    cerr << " |                                                               |"<<endl;
    cerr << " |    *** Please cite it if this program is of use to you! ***   |"<<endl;
    cerr << " -----------------------------------------------------------------"<<endl;
    cerr << "--> Maximum-likelihood multi-reference 3D-refinement"<<endl;
    if (Nvols==1)
    cerr << "  Initial reference volume : "<< fn_vol<<endl;
    else {
      cerr << "  Selfile with references  : "<<fn_vol<<endl;
      cerr << "    with # of volumes      : "<<Nvols<<endl;
    }
    cerr << "  Experimental images:     : "<< SF.name()<<" ("<<SF.ImgNo()<<")"<<endl;
    cerr << "  Angular sampling rate    : "<< angular<<endl;
    if (fn_sym!="")
    cerr << "  Symmetry file:           : "<< fn_sym<<endl;
    cerr << "  Output rootname          : "<< fn_root<<endl;
    cerr << "  Convergence criterion    : "<< eps<<endl;
    if (lowpass>0) 
    cerr << "  Low-pass filter          : "<< lowpass<<endl;
    if (mask_radius>0) 
    cerr << "  Mask radius              : "<< mask_radius<<endl;
    if (tilt_range0>0. || tilt_rangeF<90.)
    cerr << "  Limited tilt range       : "<<tilt_range0<<"  "<<tilt_rangeF<<endl;
    if (do_wbp)
    cerr << "  -> Use weighted back-projection instead of ART for reconstruction"<<endl;
    cerr << " -----------------------------------------------------------------"<<endl;

    // Also open and fill history file
    fh_hist.open((fn_root+".hist").c_str(),ios::app);
    if (!fh_hist)
      REPORT_ERROR(3008,(string)"Prog_Refine3d: Cannot open file "+fn_root+".hist");

    fh_hist << " -----------------------------------------------------------------"<<endl;
    fh_hist << " | Read more about this program in the following publication:    |"<<endl;
    fh_hist << " |  Scheres ea. (2006) in preparation ...                        |"<<endl;
    fh_hist << " |                                                               |"<<endl;
    fh_hist << " |    *** Please cite it if this program is of use to you! ***   |"<<endl;
    fh_hist << " -----------------------------------------------------------------"<<endl;
    fh_hist << "--> Maximum-likelihood multi-reference 3D-refinement"<<endl;
    fh_hist << "  Initial reference volume : "<< fn_vol<<endl;
    fh_hist << "  Experimental images:     : "<< SF.name()<<" ("<<SF.ImgNo()<<")"<<endl;
    fh_hist << "  Angular sampling rate    : "<< angular<<endl;
    if (fn_sym!="")
    fh_hist << "  Symmetry file:           : "<< fn_sym<<endl;
    fh_hist << "  Output rootname          : "<< fn_root<<endl;
    fh_hist << "  Convergence criterion    : "<< eps<<endl;
    if (lowpass>0) 
    fh_hist << "  Low-pass filter          : "<< lowpass<<endl;
    if (mask_radius>0) 
    fh_hist << "  Mask radius              : "<< mask_radius<<endl;
    if (tilt_range0>0. || tilt_rangeF<90.)
    fh_hist << "  Limited tilt range       : "<<tilt_range0<<"  "<<tilt_rangeF<<endl;
    if (do_wbp)
    fh_hist << "  -> Use weighted back-projection instead of wlsART for reconstruction"<<endl;
    fh_hist << " -----------------------------------------------------------------"<<endl;

  }

}

// Projection of the reference (blob) volume =================================
void Prog_Refine3d_prm::project_reference_volume(SelFile &SFlib, int rank) {
  
  VolumeXmipp                   vol;
  SymList                       SL;
  DocFile                       DFlib;
  FileName                      fn_proj,fn_tmp;
  Projection                    proj;
  double                        rot,tilt,psi=0.;
  int                           nvol,nl,nr_dir;

  if (fn_sym!="") SL.read_sym_file(fn_sym);
  make_even_distribution(DFlib,angular,SL,false);
  // Select use-provided tilt range
  if (tilt_range0>0. || tilt_rangeF<90.) {
    DocLine DL;
    DocFile Dt;
    DFlib.go_first_data_line();
    while (!DFlib.eof()) {
      DL=DFlib.get_current_line();
      tilt=DFlib(1);
      if (tilt>=tilt_range0 && tilt<=tilt_rangeF) Dt.append_line(DL);
      DFlib.next_data_line();	
    }
    DFlib=Dt;
    Dt.clear();
  }
  nl=Nvols*DFlib.dataLineNo();

  SFlib.clear();
  eachvol_start.clear();
  eachvol_end.clear();

  if (verb>0 && rank==0) {
    cerr << "--> projecting reference library ..."<<endl;
    init_progress_bar(nl);
  }

  // Loop over all reference volumes
  nvol=0;
  nr_dir=0;
  fn_tmp=fn_root+"_lib";
  SFvol.go_beginning();
  while (!SFvol.eof()) {
    eachvol_start.push_back(nr_dir);
    vol.read(SFvol.NextImg());
    vol().set_Xmipp_origin();
    DFlib.go_beginning();
    DFlib.adjust_to_data_line();
    while (!DFlib.eof()) {
      fn_proj.compose(fn_tmp,nr_dir+1,"proj");
      rot  = DFlib(0);
      tilt = DFlib(1);
      // In parallel case: only master actually projects and writes to disc
      if (rank==0) {
	project_Volume(vol(),proj,vol().RowNo(),vol().ColNo(),rot,tilt,psi);
	proj.set_eulerAngles(rot,tilt,psi);
	proj.write(fn_proj);
      }
      SFlib.insert(fn_proj,SelLine::ACTIVE);
      DFlib.next_data_line();
      nr_dir++;
      if (verb>0  && rank==0 && (nr_dir%MAX(1,nl/60)==0)) progress_bar(nr_dir);
    }
    eachvol_end.push_back(nr_dir-1);
    nvol++;
  }  
  if (verb>0 && rank==0) {
    progress_bar(nl);
    cerr << " -----------------------------------------------------------------"<<endl;
  }

  if (rank==0) {
    fn_tmp=fn_root+"_lib.sel";
    SFlib.write(fn_tmp);
    fn_tmp=fn_root+"_lib.doc";
    DFlib.write(fn_tmp);
  }

  // Free memory
  vol.clear();

}

// Reconstruction using the ML-weights ==========================================
void Prog_Refine3d_prm::reconstruction(int argc, char **argv, 
				       int iter, int volno) {

  VolumeXmipp            new_vol;
  FileName               fn_tmp,fn_insel,fn_blob;
  SelFile                SFall,SFone;

  fn_tmp=fn_root+"_it";
  fn_tmp.compose(fn_tmp,iter,"");
  if (iter>1) {
    fn_blob=fn_root+"_it";
    fn_blob.compose(fn_blob,iter-1,"");
  } else fn_blob="";

  // Setup selfile for reconstruction
  fn_insel=fn_tmp+".sel";
  if (Nvols>1) {
    fn_tmp+="_vol";
    fn_tmp.compose(fn_tmp,volno+1,"");
    if (fn_blob!="") {
      fn_blob+="_vol";
      fn_blob.compose(fn_blob,volno+1,"basis");
    }
    // Select only relevant projections to reconstruct
    SFall.read(fn_insel);
    SFall.go_beginning();
    for (int nr=eachvol_start[volno]; nr<=eachvol_end[volno]; nr++) {
      SFall.go_beginning();
      SFall.jump_lines(nr);
      SFone.insert(SFall.current());
    }
    fn_insel=fn_tmp+".sel";
    SFone.write(fn_insel);
  } else {
    if (fn_blob!="") fn_blob+=".basis";
  }

  if (!do_wbp) {
    Basic_ART_Parameters   art_prm;
    Plain_ART_Parameters   dummy;
    GridVolume             new_blobs;
    GridVolume             start_blobs;
    if (verb>0) cerr << "--> weighted least-squares ART reconstruction "<<endl;

    // Read ART parameters from command line & I/O with outer loop of Refine3d
    art_prm.read(argc, argv);
    art_prm.WLS=true;
    if (!check_param(argc,argv,"-n")) 
      art_prm.no_it=10;
    if (!check_param(argc,argv,"-l")) {
      art_prm.lambda_list.resize(1); 
      art_prm.lambda_list.init_constant(0.2);
    }
    if (!check_param(argc,argv,"-k")) {
      art_prm.kappa_list.resize(1); 
      art_prm.kappa_list.init_constant(0.5);
    }
    art_prm.fn_sel=fn_insel;
    art_prm.fn_root=fn_tmp;
    art_prm.tell=TELL_SAVE_BASIS;
    art_prm.fn_start=fn_blob;
    // Reconstruct using weighted least-squares ART
    Basic_ROUT_Art(art_prm,dummy,new_vol,new_blobs);

  } else {

    Prog_WBP_prm           wbp_prm;
    if (verb>0) cerr << "--> WBP reconstruction "<<endl;

    // read command line (fn_sym, angular etc.)
    wbp_prm.read(argc, argv);
    if (mask_radius>0) wbp_prm.diameter=(int)mask_radius*2;
    wbp_prm.fn_sel=fn_insel;
    wbp_prm.do_weights=true;
    wbp_prm.do_all_matrices=true;
    wbp_prm.show();
    wbp_prm.verb=verb;
    wbp_prm.fn_out=fn_tmp+".vol";
    wbp_prm.produce_Side_info();
    wbp_prm.apply_2Dfilter_arbitrary_geometry(wbp_prm.SF,new_vol);
    new_vol.write(wbp_prm.fn_out);
  }

  if (verb>0) cerr << " -----------------------------------------------------------------"<<endl;

}

void Prog_Refine3d_prm::remake_SFvol(int iter, bool rewrite) {

  FileName               fn_tmp,fn_tmp2;
  int                    volno=0;
  VolumeXmipp            ref_vol;

  fn_tmp=fn_root+"_it";
  fn_tmp.compose(fn_tmp,iter,"");

  // Initial iteration: copy volumes to correct name for iteration
  // loop, and rewrite with this name to disc
  if (rewrite) {
    SFvol.go_beginning();
    while (!SFvol.eof()) {
      ref_vol.read(SFvol.NextImg());
      ref_vol().set_Xmipp_origin();
      if (Nvols>1) {
	fn_tmp2=fn_tmp+"_vol";
	fn_tmp2.compose(fn_tmp2,volno+1,"vol");
      } else fn_tmp2=fn_tmp+".vol";
      ref_vol.write(fn_tmp2);
      volno++;
    }
  }
  
  // Update selection file for reference volumes
  SFvol.clear();
  if (Nvols>1) {
    fn_tmp+="_vol";
    volno=0;
    while (volno<Nvols) {
      fn_tmp2.compose(fn_tmp,volno+1,"vol");
      SFvol.insert(fn_tmp2);
      volno++;
    } 
  } else {
    SFvol.insert(fn_tmp+".vol");
  }

}


// Modify reference volume ======================================================
void Prog_Refine3d_prm::post_process_volumes(int argc, char **argv) {

  Mask_Params            mask_prm;
  FileName               fn_vol,fn_tmp;
  VolumeXmipp            vol,Vaux;
  SymList                SL;
  matrix3D<int>          mask3D;
  double                 avg,dummy;
  int                    dim;

  if ( (fn_sym!="") || (fn_solv!="") || (mask_radius>0) || (lowpass>0) ) {
    SFvol.go_beginning();
    while (!SFvol.eof()) {
      // Read corresponding volume from disc
      fn_vol=SFvol.NextImg();
      vol.read(fn_vol);
      vol().set_Xmipp_origin();
      dim=vol().RowNo();
      // Store the original volume on disc
      fn_tmp=fn_vol+".original";
      vol.write(fn_tmp);

      // Symmetrize if requested
      if (fn_sym!="") {
	Vaux().resize(vol());
	SL.read_sym_file(fn_sym);
	symmetrize(SL,vol,Vaux);
	vol=Vaux;
	Vaux.clear();
      }

      // Solvent flattening if requested
      if (fn_solv!="") {
	VolumeXmipp solv;
	solv.read(fn_solv);
	solv()=1.-solv();
	double solvavg=0.,sumsolv=0.;
	FOR_ALL_DIRECT_ELEMENTS_IN_MATRIX3D(vol()) {
	  solvavg+=dVkij(solv(),k,i,j)*dVkij(vol(),k,i,j);
	  sumsolv+=dVkij(solv(),k,i,j);
	}
	solvavg/=sumsolv;
	FOR_ALL_DIRECT_ELEMENTS_IN_MATRIX3D(solv()) {
	  dVkij(vol(),k,i,j)-=dVkij(solv(),k,i,j)*(dVkij(vol(),k,i,j)-solvavg);
	}
      }

      // Masking out the parts outside the inner sphere
      if (mask_radius>0) {
	mask_prm.mode=OUTSIDE_MASK;
	mask_prm.R1=mask_radius;
	mask_prm.type=BINARY_CIRCULAR_MASK;
	mask_prm.generate_3Dmask(vol());
	compute_stats_within_binary_mask(mask_prm.imask3D,vol(),dummy,dummy,avg,dummy);
	mask_prm.mode=INNER_MASK;
	mask_prm.generate_3Dmask(vol());
	mask_prm.apply_mask(vol(),vol(),avg);
      }

      // Filtering the volume
      if (lowpass>0) {
	FourierMask fmask;
	fmask.raised_w=0.02;
	fmask.FilterShape=RAISED_COSINE;
	fmask.FilterBand=LOWPASS;
	fmask.w1=lowpass;
	fmask.apply_mask_Space(vol());
      }

      // (Re-) write post-processed volume to disc
      vol.write(fn_vol);
      
    }
    if (verb>0) cerr << " -----------------------------------------------------------------"<<endl;
  }

}

// Convergence check ===============================================================
bool Prog_Refine3d_prm::check_convergence(int iter)  {

  VolumeXmipp            vol,old_vol, diff_vol;
  FileName               fn_tmp;
  Mask_Params            mask_prm;
  matrix3D<int>          mask3D;
  double                 signal,change;
  int                    volno=0,dim;
  bool                   converged=true;

  if (verb>0) cerr << "--> checking convergence "<<endl;

  SFvol.go_beginning();
  while (!SFvol.eof()) {
    // Read corresponding volume from disc
    fn_vol=SFvol.NextImg();
    vol.read(fn_vol);
    vol().set_Xmipp_origin();
    dim=vol().RowNo();
    old_vol().init_zeros(vol());
    diff_vol().init_zeros(vol());

    // Only consider voxels within the spherical mask
    mask_prm.R1=dim/2;
    mask_prm.type=BINARY_CIRCULAR_MASK;
    mask_prm.mode=INNER_MASK;
    mask_prm.generate_3Dmask(vol());
    fn_tmp=fn_root+"_it";
    fn_tmp.compose(fn_tmp,iter-1,"");
    if (Nvols>1) {
      fn_tmp+="_vol";
      fn_tmp.compose(fn_tmp,volno+1,"vol");
    } else fn_tmp+=".vol";
    old_vol.read(fn_tmp);
    diff_vol()=vol()-old_vol();
    mask_prm.apply_mask(old_vol(),old_vol());
    mask_prm.apply_mask(diff_vol(),diff_vol());
    change=diff_vol().sum2();
    signal=old_vol().sum2();
    if (change/signal>eps) converged=false;
    if (verb>0) {
      if (Nvols>1) {
	cerr << "Relative signal change volume "<<volno+1<<" = " <<change/signal<<endl;
	fh_hist << "Relative signal change volume "<<volno+1<<" = " <<change/signal<<endl;
      } else {
	cerr << "Relative signal change volume = " <<change/signal<<endl;
	fh_hist << "Relative signal change volume = " <<change/signal<<endl;
      }
    }
    volno++;
  }

  return converged;

}
