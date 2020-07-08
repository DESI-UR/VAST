#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <iostream>
#include <vector>

using namespace std;

#define ngrid 128
#define ngalmax 1000000
#define nmaxvd 2000000
#define chainmax 500
#define ntot 1000000
#define ngroupmax 100000
#define maskra 360
#define maskdec 180
#define decoffset -90
#define maxdist 300.   //z = 0.107 -> 313 h-1 mpc   z = 0.087 -> 257 h-1 mpc
#define mindist 0.

int max(int in1, int in2) {
  if (in1>in2)
    return in1;
  return in2;
}

int min(int in1, int in2) {
  if (in1<in2)
    return in1;
  return in2;
}

void qsort(vector<vector<double> > &myvoids, int nvd) {

  if (nvd <= 1)
    return;
  vector<vector<double> > less;
  vector<vector<double> > more;
  vector<double> pivot = myvoids[0];
  int i;
  int nless = 0;
  int nmore = 0;

  for (i=1;i<nvd;i++) {
    if (myvoids[i][3]>pivot[3]) {
      more.push_back(myvoids[i]);
      nmore++;
    }
    else {
      less.push_back(myvoids[i]);
      nless++;
    }
  }
  qsort(less,nless);
  qsort(more,nmore);

  for (i=0;i<nmore;i++) {
    myvoids[i] = more[i];
  }
  myvoids[nmore] = pivot;
  for (i=0;i<nless;i++) {
    myvoids[nmore+i+1] = less[i];
  }

}

int sign(double in) {
  if (in>0)
    return 1;
  if (in<0)
    return -1;
  return 0;
}

double dist(double x1, double y1, double z1, double x2, double y2, double z2) {
  return sqrt(pow(x2-x1,2)+pow(y2-y1,2)+pow(z2-z1,2));
}

int main() {   

  printf("predecs\n");

  FILE *infile, *out1, *out2, *out3, *voidgals, *maskfile;
  int nv,p,nvd,ntouch;
  int nin,nintest;
  
  vector<int> vtemp1(chainmax,0);
  vector<vector<int> > vtemp2(chainmax,vtemp1);
  vector<vector<vector<int> > > chainlist(chainmax,vtemp2);
  vector<int> linklist(ngalmax,0);
  vector<bool> btemp1(maskdec,0);
  vector<vector<bool> > mask(maskra,btemp1);
  
  int nwall,nf;
  vector<vector<vector<int> > > tempchain(chainmax,vtemp2);
  int meshi,meshj,meshk;
  vector<vector<vector<int> > > ngal(chainmax,vtemp2);
  int stopat,jgal,igx,igy,igz;
  int i,j,k,k1g,nposs,ioff,igal;
  int ncell,k2g,k3g,ixc,iyc,izc,k4g;
  int ncheck,ncheck2,flg,k4g1,k4g2;
  vector<int> flag(ngalmax,0);
  vector<int> flog(ngalmax,0);
  vector<int> nfield(ngalmax,0);
  vector<int> flagg(ngalmax,0);
  int nsph,nran;
  vector<int> n(ngroupmax,0);
  int ngroup,nspecial;
  int npv,nh,mi,mj,mk,ioff2,count;
  int skiptot;

  printf("integer decls\n");
  double l,sd,var,avsep;
  vector<double> minsep2(ngalmax,0),minsep1(ngalmax,0),minsep3(ngalmax,0);
  double totsep,bla;
  double a1,a2,v,vs,r1,r2,c,voidmax;

  vector<double> vtemp3(3,0);
  vector<double> vtemp4(4,0);
  //vector<double> galin(ngalmax,vtemp3);
  //vector<double> galw(ngalmax,vtemp3);
  //vector<double> galv(ngalmax,vtemp4);
  //vector<double> galf(ngalmax,vtemp3);
  vector<vector<double> > myvoids(nmaxvd,vtemp4);
  //vector<double> holes(nmaxvd,vtemp4);

  vector<double> xin(ngalmax,0),yin(ngalmax,0),zin(ngalmax,0);
  vector<double> xw(ngalmax,0),yw(ngalmax,0),zw(ngalmax,0);
  vector<double> xv(ngalmax,0),yv(ngalmax,0),zv(ngalmax,0),rv(ngalmax,0);
  vector<double> xf(ngalmax,0),yf(ngalmax,0),zf(ngalmax,0);
  vector<double> xvd(nmaxvd,0),yvd(nmaxvd,0),zvd(nmaxvd,0),rvd(nmaxvd,0);
  vector<double> xh(nmaxvd,0),yh(nmaxvd,0),zh(nmaxvd,0),rh(nmaxvd,0);

  double vol,dl,box,minsep,near,zdiff,ydiff,xdiff;
  double xcen,ycen,zcen,rad,sep,zcen2,ycen2,xcen2;
  double minx,modv;
  vector<double> voidvol(ngalmax,0);
  double x2,top,bot,BAx,BAy,BAz,vx,vy,vz,test;
  double rad2,xcen3,ycen3,zcen3,xbi,ybi,zbi;
  double CEx,CEy,CEz,ACx,ACy,ACz;
  double rad3,dotu2,dotv2,xcen4,ycen4,zcen4;
  double wx,wy,wz,modw,ux,uy,uz,modu;
  double pi,ran3,seed,diff;
  double minx1,minx2,frac,mr;
  double xcen41,ycen41,zcen41,xcen42,ycen42,zcen42;
  double dotu1,dotv1,x,y,z,deltap,r;
  double absmag,dosep,xmin,xmax,ymin,ymax,zmin,zmax;
  double raf,decf,rtod,dtor,rav,decv,ra,dec;
  double temp;

  //EDIT FILE NAMES HERE
  //Inputs
  infile = fopen("../data/SDSS/vollim_dr7_cbp_102709.dat","r"); // File format: RA, dec, redshift, comoving distance, absolute magnitude
  maskfile = fopen("../data/SDSS/cbpdr7mask.dat","r"); // File format: RA, dec
  //Outputs
  out1 = fopen("../void_catalogs/SDSS/Cpp_implementation/out1_vollim_dr7_cbp_102709.dat","w"); // List of maximal spheres of each void region: x, y, z, radius, distance, ra, dec
  out2 = fopen("../void_catalogs/SDSS/Cpp_implementation/out2_vollim_dr7_cbp_102709.dat","w"); // List of holes for all void regions: x, y, z, radius, flog (to which void it belongs)
  out3 = fopen("../void_catalogs/SDSS/Cpp_implementation/out3_vollim_dr7_cbp_102709.dat","w"); // List of void region sizes: radius, effective radius, evolume, x, y, z, deltap, nfield, vol_maxhole
  voidgals = fopen("../void_catalogs/SDSS/Cpp_implementation/vollim_voidgals_dr7_cbp_102709.dat","w"); // List of the void galaxies: x, y, z, void region #
  frac = 0.1;
  dosep = 1;

  //CONSTANTS
  c = 300000.;
  pi = 4.0*atan(1.0);
  dtor = 0.0174532925;
  rtod = 57.29577951;

  //INITIALIZATIONS
  npv = 0;
  seed = -78197341;
  box = 630.;
  vol = 0;
  dl = box/(double)ngrid;
  voidmax = 100.;
  ioff2 = (int)(voidmax/dl) + 2;

  printf("number of grid cells is %d %f %f %f\n",ngrid,dl,box,ioff2);
  ncell = 0;

  //read mask
  while (fscanf(maskfile,"%lf %lf\n",&ra,&dec)!=EOF) {
    mask[int(ra)][int(dec)-decoffset] = 1;
    vol++;
  }
  printf("read mask\n");


  //read in the x,y,z positions in the simulation
  nin = 0;
  xmin = 1000.;
  ymin = 1000.;
  zmin = 1000.;

  xmax = -1000.;
  ymax = -1000.;
  zmax = -1000.;  

  while (fscanf(infile,"%lf %lf %lf %lf %lf\n",&ra,&dec,&z,&r,&absmag)!=EOF) {
    xin[nin] = r*cos(ra*dtor)*cos(dec*dtor);
    yin[nin] = r*sin(ra*dtor)*cos(dec*dtor);
    zin[nin] = r*sin(dec*dtor);
    xmin = min(xmin,xin[nin]);
    ymin = min(ymin,yin[nin]);
    zmin = min(zmin,zin[nin]);
    xmax = max(xmax,xin[nin]);
    ymax = max(ymax,yin[nin]);
    zmax = max(zmax,zin[nin]);
    nin++;
  }

  printf("x: %f %f\n",xmin,xmax);
  printf("y: %f %f\n",ymin,ymax);
  printf("z: %f %f\n",zmin,zmax);

  printf("There are %d galaxies in this simulation\n",nin);

  //put the galaxies on a chain mesh
  for (meshi=0;meshi<ngrid;meshi++) {
    for (meshj=0;meshj<ngrid;meshj++) {
      for (meshk=0;meshk<ngrid;meshk++) {
	      chainlist[meshi][meshj][meshk] = -1;
	      tempchain[meshi][meshj][meshk] = 0;
      }
    }
  }

  printf("making the grid\n");
  for (igal=0;igal<nin;igal++) {
    meshi = (int)((xin[igal]-xmin)/dl);
    meshj = (int)((yin[igal]-ymin)/dl);
    meshk = (int)((zin[igal]-zmin)/dl);
    ngal[meshi][meshj][meshk]++;
    linklist[igal] = chainlist[meshi][meshj][meshk];
    chainlist[meshi][meshj][meshk] = igal;
  }

  printf("made the grid\n");
  printf("checking the grid\n");
  for (i=0;i<ngrid;i++) {
    for (j=0;j<ngrid;j++) {
      for (k=0;k<ngrid;k++) {
	      count = 0;
	      igal = chainlist[i][j][k];
	      while (igal!=-1) {
	        count++;
	        igal = linklist[igal];
	      }
	      if (count!=ngal[i][j][k]) {
	        printf("%d %d %d %d %d\n",i,j,k,count,ngal[i][j][k]);
	      }
      }
    }
  }

  nintest = nin;

  printf("finding sep\n");
  totsep = 0.;
  for (i=0;i<nintest;i++) {
    minsep1[i] = 10000000.;
    minsep2[i] = 10000000.;
    minsep3[i] = 10000000.;
    if (!(i%10000))
      printf("%d\n",i);
    mi = (int)((xin[i]-xmin)/dl);
    mj = (int)((yin[i]-ymin)/dl);
    mk = (int)((zin[i]-zmin)/dl);
    ioff = 0;
    nposs = 0;
    while (nposs<6) {
      ioff++;
      nposs = 0;
      for (meshi=max(0,mi-ioff);meshi<min(ngrid,mi+ioff);meshi++) {
	      for (meshj=max(0,mj-ioff);meshj<min(ngrid,mj+ioff);meshj++) {
	        for (meshk=max(0,mk-ioff);meshk<min(ngrid,mk+ioff);meshk++) {
	          nposs+=ngal[meshi][meshj][meshk];
	        }
	      }
      }
    }
    //now need to check that there isn't one closer in shell just round it
    ioff++;
    for (meshi=max(0,mi-ioff);meshi<min(ngrid,mi+ioff);meshi++) {
      for (meshj=max(0,mj-ioff);meshj<min(ngrid,mj+ioff);meshj++) {
        for (meshk=max(0,mk-ioff);meshk<min(ngrid,mk+ioff);meshk++) {
	        jgal = chainlist[meshi][meshj][meshk];
	        while (jgal!=-1) {
	          sep = pow(xin[i]-xin[jgal],2)+pow(yin[i]-yin[jgal],2)+pow(zin[i]-zin[jgal],2);
	          if (sep>0) {
	            if (sep<minsep1[i]) {
		          minsep3[i] = minsep2[i];
		          minsep2[i] = minsep1[i];
		          minsep1[i] = sep;
	            }
	            else if (sep<minsep2[i]) {
	              minsep3[i] = minsep2[i];
	              minsep2[i] = sep;
	            }
	            else if (sep<minsep3[i]) {
	              minsep3[i] = sep;
	            }
	          }
	          jgal = linklist[jgal];
	        }
	      }
      }
    }
    minsep1[i] = sqrt(minsep1[i]);
    minsep2[i] = sqrt(minsep2[i]);
    minsep3[i] = sqrt(minsep3[i]);
    totsep = minsep3[i] + totsep;
  }
  avsep = totsep/(double)(nintest);
  printf("average separation of n3rd gal is %f\n",avsep);
  var = 0.;
  sd = 0.;
  for (i=0;i<nintest;i++) {
    var += pow(minsep3[i]-avsep,2);
  }
  sd = sqrt(var/(double)(nintest));
  printf("the standard deviation is %f\n",sd);
  l = avsep + 1.5*sd;
  // l = 5.81637;  //s0 = 7.8, gamma = 1.2, void edge = -0.8
  // l = 7.36181;  //s0 = 3.5, gamma = 1.4
  // or force l to haev a fixed number by setting l = ****

  printf("going to build wall with search value %f\n",l);

  nf = 0;
  nwall = 0;
  for (i=0;i<nin;i++) {
    if (minsep3[i]>l) {
      nf++;
      xf[nf] = xin[i];
      yf[nf] = yin[i];
      zf[nf] = zin[i];
      //fprintf(fieldgals,"%f %f %f\n",xf[nf],yf[nf],zf[nf]);
    }
    else {
      nwall++;
      xw[nwall] = xin[i];
      yw[nwall] = yin[i];
      zw[nwall] = zin[i];
      //fprintf(wallgals,"%f %f %f\n",xw[nwall],yw[nwall],zw[nwall]);
    }
  }
  printf("%d %d\n",nf,nwall);

  //set up the cell grid distribution
  for (i=0;i<ngrid;i++) {
    for (j=0;j<ngrid;j++) {
      for (k=0;k<ngrid;k++) {
	      ngal[i][j][k]=0;
      }
    }
  }

  for (i=0;i<nwall;i++) {
    igx = (int)((xw[i]-xmin)/dl);
    igy = (int)((yw[i]-ymin)/dl);
    igz = (int)((zw[i]-zmin)/dl);
    ngal[igx][igy][igz]++;
  }

  //Sort the galaxies into a chaining mesh.
  printf("Constructing galaxy chaining mesh\n");
  for (meshi=0;meshi<ngrid;meshi++) {
    for (meshj=0;meshj<ngrid;meshj++) {
      for (meshk=0;meshk<ngrid;meshk++) {
	      chainlist[meshi][meshj][meshk]=-1;
	      tempchain[meshi][meshj][meshk]=0;
      }
    }
  }

  for (igal=0;igal<nwall;igal++) {
    meshi=(int)((xw[igal]-xmin)/dl);
    meshj=(int)((yw[igal]-ymin)/dl);
    meshk=(int)((zw[igal]-zmin)/dl);
    if (chainlist[meshi][meshj][meshk]==-1) {
      chainlist[meshi][meshj][meshk]=igal;
      tempchain[meshi][meshj][meshk]=igal;
    }
    else {
      linklist[tempchain[meshi][meshj][meshk]]=igal;
      tempchain[meshi][meshj][meshk]=igal;
    }
  }
 
  for (meshi=0;meshi<ngrid;meshi++) {
    for (meshj=0;meshj<ngrid;meshj++) {
      for (meshk=0;meshk<ngrid;meshk++) {
	      if (chainlist[meshi][meshj][meshk]!=-1) {
	        linklist[tempchain[meshi][meshj][meshk]]=chainlist[meshi][meshj][meshk];
	      }
      }
    }
  }

  //first find an empty cell

  nvd = 0;
  skiptot = 0;
  //  for (i=19;i<20;i++) {
  for (i=0;i<ngrid;i++) {
    printf("%d\n",i);
    //    for (j=39;j<40;j++) {
    for (j=0;j<ngrid;j++) {
      //      printf("%d %d\n",i,j);
      for (k=0;k<ngrid;k++) {
	    xcen = (i+0.5)*dl+xmin;
	    ycen = (j+0.5)*dl+ymin;
	    zcen = (k+0.5)*dl+zmin;

	    r = sqrt(pow(xcen,2)+pow(ycen,2)+pow(zcen,2));
	    ra = atan(ycen/xcen)*180./pi;
	    dec = asin(zcen/r)*180./pi;
	    if ((ycen>0)&&(xcen<0)) 
	      ra += 180.;
	    if ((ycen<0)&&(xcen<0)) 
	      ra += 180.;
	    if (ra<0.)
	      ra += 360.;
	    //printf("%f %f %f %f %f\n",xcen,ycen,zcen,ra,dec);
	    if ((mask[int(ra)][int(dec)-decoffset]==0)||(r>maxdist)) {
	      //printf("skipped %d %d\n",int(ra),int(dec));
	      continue;
	    }

	    if (ngal[i][j][k]==0) { //this is an empty cell and worth growing a void from it
	      //	  printf("Empty cell at: %d %d %d\n",i,j,k);
	      ioff = 0;
	      nposs = 0;
	      while (nposs<1) {
	        ioff++;
	        for (meshi=max(0,i-ioff);meshi<min(ngrid,i+ioff);meshi++) {
	          xdiff = (meshi+0.5)*dl;
	          for (meshj=max(0,j-ioff);meshj<min(ngrid,j+ioff);meshj++) {
		        ydiff = (meshj-0.5)*dl;
		        for (meshk=max(0,k-ioff);meshk<min(ngrid,k+ioff);meshk++) {
		          zdiff = (meshk-0.5)*dl;
		          nposs+=ngal[meshi][meshj][meshk];
		          //		  printf("%d %d %d %d\n",meshi,meshj,meshk,ngal[meshi][meshj][meshk]);
		        }
	          }
	        }
	      }
	      //printf("past nposs\n");
	      // now need to check that there isn't one closer in shell just round it
	      ioff++;
	      minsep = 100000.;
	      for (meshi=max(0,i-ioff);meshi<min(ngrid,i+ioff);meshi++) {
	        xdiff = (meshi+0.5)*dl;
	        for (meshj=max(0,j-ioff);meshj<min(ngrid,j+ioff);meshj++) {
	          ydiff = (meshj+0.5)*dl;
	          for (meshk=max(0,k-ioff);meshk<min(ngrid,k+ioff);meshk++) {
		        zdiff = (meshk+0.5)*dl;
		        jgal = chainlist[meshi][meshj][meshk];
		        //  printf("meshi meshj meshk chainlist, %d %d %d %d\n",meshi,meshj,meshk,jgal);
		        if (jgal!=-1) {
		          stopat=-1;
		          //   printf("jgal linklist[jgal]: %d %d\n",jgal,linklist[jgal]);
		          jgal = linklist[jgal];
		          while (jgal!=stopat) {
		            if (stopat==-1)
		              stopat = jgal;
		            near = pow(xcen-xw[jgal],2)+pow(ycen-yw[jgal],2)+pow(zcen-zw[jgal],2);
		            if (near<minsep) {
		              minsep = near;
		              k1g = jgal;
		            }
		            jgal = linklist[jgal];
		            //	printf("jgal stopat, %d %d\n",jgal, stopat);
		          }
		        }
	          }
	        }
	      }
	      //printf("Found closest gal\n");
	      //now found the first galaxy that is closest to the center of the empty cell
	      modv = sqrt(pow(xcen-xw[k1g],2)+pow(ycen-yw[k1g],2)+pow(zcen-zw[k1g],2));
	      vx = (xw[k1g]-xcen)/modv;
	      vy = (yw[k1g]-ycen)/modv;
	      vz = (zw[k1g]-zcen)/modv;
	      //Andrew's way at getting the next galaxy works.
	      minx = 10000;
	      k2g = 0;
	      //want to find the next closest galaxy.  Minimize x = top/bottom
	      //again need to find ioff and look in ioff + 1 cells to find this
	      //still based in first cell here
	      //however, next nearest galaxy can lie ANYWHERE on vector that
	      //joins Xcen1 and Gal1.
	      //changing this - run a coarse grid first to find the largest void
	      //then use that (plus a little slack) to restrict how many
	      //particles you need to look at.
	      ioff = min(ngrid,ioff2);
	      minsep = 100000.;
	      for (meshi=max(0,i-ioff);meshi<min(ngrid,i+ioff);meshi++) {
	        xdiff = (meshi+0.5)*dl;
	        for (meshj=max(0,j-ioff);meshj<min(ngrid,j+ioff);meshj++) {
	          ydiff = (meshj+0.5)*dl;
	          for (meshk=max(0,k-ioff);meshk<min(ngrid,k+ioff);meshk++) {
		        zdiff = (meshk+0.5)*dl;
		        jgal = chainlist[meshi][meshj][meshk];
		        if (jgal!=-1) {
		          stopat = -1;
		          jgal = linklist[jgal];
		          while (jgal!=stopat) {
		            if (stopat==-1)
		              stopat = jgal;
		            if (jgal!=k1g) {
		              BAx = xw[k1g] - xw[jgal];
		              BAy = yw[k1g] - yw[jgal];
		              BAz = zw[k1g] - zw[jgal];
		              bot = 2*(BAx*vx + BAy*vy + BAz*vz);
		              top = pow(BAx,2)+pow(BAy,2)+pow(BAz,2);
		              x2 = top/bot;
		              if ((x2>0)&&(x2<minx)) {
			            minx = x2;
			            k2g = jgal;
		              }
		            }
		            jgal = linklist[jgal];
		          }
		        }
	          }
	        }
	      }
	      x2 = minx;
	      if (minx<10000.) {
	        //this is a problem - not possible to constrain void,
	        //probably at edge of survey
	        //do not consider this center further

	        //printf("found 2nd gal\n");

	        xcen2 = xw[k1g] - x2*vx;
	        ycen2 = yw[k1g] - x2*vy;
	        zcen2 = zw[k1g] - x2*vz;

	        r = sqrt(pow(xcen2,2)+pow(ycen2,2)+pow(zcen2,2));
	        ra = atan(ycen2/xcen2)*180./pi;
	        dec = asin(zcen2/r)*180./pi;
	        if ((ycen2>0)&&(xcen2<0)) 
	          ra += 180.;
	        if ((ycen2<0)&&(xcen2<0)) 
	          ra += 180.;
	        if (ra<0.)
	          ra += 360.;
	        if ((mask[int(ra)][int(dec)-decoffset]==0)||(r>maxdist)||(r<mindist))
	          continue;

	        sep = sqrt(pow(xcen2-xw[k1g],2)+pow(ycen2-yw[k1g],2)+pow(zcen2-zw[k1g],2));
	        rad = sqrt(pow(xcen2-xw[k2g],2)+pow(ycen2-yw[k2g],2)+pow(zcen2-zw[k2g],2));

	        //these must be in the sim
	        if ((xcen2>xmin)&&(xcen2<xmax)&&(ycen2>ymin)&&(ycen2<ymax)&&(zcen2>zmin)&&(zcen2<zmax)) {
	          //now found second galaxy, have to find the third. 
	          //Now need to know which cell the new center is in though.
	          ixc = (int)((xcen2-xmin)/dl);
	          iyc = (int)((ycen2-ymin)/dl);
	          izc = (int)((zcen2-zmin)/dl);
	          //these are bisection parts
	          xbi = (xw[k1g]+xw[k2g])/2;
	          ybi = (yw[k1g]+yw[k2g])/2;
	          zbi = (zw[k1g]+zw[k2g])/2;
	          //need to find the bit I need to add on to (xcen2,ycen2,zcen2)
	          //work out unit vector
	          modv = sqrt(pow(xcen2-xbi,2)+pow(ycen2-ybi,2)+pow(zcen2-zbi,2));
	          vx = (xcen2-xbi)/modv;
	          vy = (ycen2-ybi)/modv;
	          vz = (zcen2-zbi)/modv;
	          minx = 100000;
	          //very similar piece of code to that above
	          //but center box is where new center is
	          //now need to check that isn't one closer in 2 shells just round it
	          for (meshi=max(0,ixc-ioff);meshi<min(ngrid,ixc+ioff);meshi++) {
		        for (meshj=max(0,iyc-ioff);meshj<min(ngrid,iyc+ioff);meshj++) {
		          for (meshk=max(0,izc-ioff);meshk<min(ngrid,izc+ioff);meshk++) {
		            jgal = chainlist[meshi][meshj][meshk];
		            if (jgal!=-1) {
		              stopat = -1;
		              jgal = linklist[jgal];
		              while (jgal!=stopat) {
			            if (stopat == -1)
			              stopat = jgal;
			            if ((jgal!=k1g)&&(jgal!=k2g)) {
			              sep = sqrt(pow(xw[jgal]-xcen2,2)+pow(yw[jgal]-ycen2,2)+pow(zw[jgal]-zcen2,2));
			              ACx = xw[k1g] - xcen2;
			              ACy = yw[k1g] - ycen2;
			              ACz = zw[k1g] - zcen2;
			              top = pow(sep,2)-pow(rad,2);
			              CEx = xw[jgal] - xcen2;
			              CEy = yw[jgal] - ycen2;
			              CEz = zw[jgal] - zcen2;
			              bot = 2*(vx*CEx+vy*CEy+vz*CEz-vx*ACx-vy*ACy-vz*ACz);
			              x2 = top/bot;
			              if ((x2>0)&&(x2<minx)) {
			                minx = x2;
			                k3g = jgal;
			              }
			            }
			            jgal = linklist[jgal];
		              }
		            }
		          }
		        }
	          }
	          if (minx<10000) {

		        //	printf("Found 3rd gal\n");

		        xcen3 = xcen2+minx*vx;
		        ycen3 = ycen2+minx*vy;
		        zcen3 = zcen2+minx*vz;

		        r = sqrt(pow(xcen3,2)+pow(ycen3,2)+pow(zcen3,2));
		        ra = atan(ycen3/xcen3)*180./pi;
		        dec = asin(zcen3/r)*180./pi;
		        if ((ycen3>0)&&(xcen3<0)) 
		          ra += 180.;
		        if ((ycen3<0)&&(xcen3<0)) 
		          ra += 180.;
		        if (ra<0.)
		          ra += 360.;
		        if ((mask[int(ra)][int(dec)-decoffset]==0)||(r>maxdist)||(r<mindist))
		          continue;

		        sep = sqrt(pow(xcen3-xw[k1g],2)+pow(ycen3-yw[k1g],2)+pow(zcen3-zw[k1g],2));
		        rad = sqrt(pow(xcen3-xw[k2g],2)+pow(ycen3-yw[k2g],2)+pow(zcen3-zw[k2g],2));
		        rad2 = sqrt(pow(xcen3-xw[k3g],2)+pow(ycen3-yw[k3g],2)+pow(zcen3-zw[k3g],2));
		        if ((xcen3>xmin)&&(xcen3<xmax)&&(ycen3>ymin)&&(ycen3<ymax)&&(zcen3>zmin)&&(zcen3<zmax)) {
		          //and finally find the fourth galaxy - bit more involved but
		          //same idea, don't know if have to mvoe above or below plane
		          ixc = (int)((xcen3-xmin)/dl);
		          iyc = (int)((ycen3-ymin)/dl);
		          izc = (int)((zcen3-zmin)/dl);
		          //this is the first run through
		          modu = sqrt(pow(xw[k1g]-xw[k2g],2)+pow(yw[k1g]-yw[k2g],2)+pow(zw[k1g]-zw[k2g],2));
		          ux = (xw[k1g]-xw[k2g])/modv;
		          uy = (yw[k1g]-yw[k2g])/modv;
		          uz = (zw[k1g]-zw[k2g])/modv;
		          modv = sqrt(pow(xw[k2g]-xw[k3g],2)+pow(yw[k2g]-yw[k3g],2)+pow(zw[k2g]-zw[k3g],2));
		          vx = (xw[k3g]-xw[k2g])/modu;
		          vy = (yw[k3g]-yw[k2g])/modu;
		          vz = (zw[k3g]-zw[k2g])/modu;
		          //new vector is
		          wx = (uy*vz-uz*vy);
		          wy = (-ux*vz+uz*vx);
		          wz = (ux*vy-uy*vx);
		          modw = sqrt(pow(wx,2)+pow(wy,2)+pow(wz,2));
		          wx/=modw;
		          wy/=modw;
		          wz/=modw;
		          //now as before.  Move on wx,wy,wz from
		          minx1 = 100000.;
		          //very similar piece of code to that above but center box
		          //is where new center is
		          for (meshi=max(0,ixc-ioff);meshi<min(ngrid,ixc+ioff);meshi++) {
		            for (meshj=max(0,iyc-ioff);meshj<min(ngrid,iyc+ioff);meshj++) {
		              for (meshk=max(0,izc-ioff);meshk<min(ngrid,izc+ioff);meshk++) {
			            jgal = chainlist[meshi][meshj][meshk];
			            if (jgal!=-1) {
			              stopat = -1;
			              jgal = linklist[jgal];
			              while (jgal!=stopat) {
			                if (stopat==-1)
			                  stopat = jgal;
			                if ((jgal!=k1g)&&(jgal!=k2g)&&(jgal!=k3g)) {
			                  sep = sqrt(pow(xw[jgal]-xcen3,2)+pow(yw[jgal]-ycen3,2)+pow(zw[jgal]-zcen3,2));
			                  ACx = xw[k1g]-xcen3;
			                  ACy = yw[k1g]-ycen3;
			                  ACz = zw[k1g]-zcen3;
			      
			                  top = pow(sep,2) - pow(rad,2);
			      
			                  CEx = xw[jgal]-xcen3;
			                  CEy = yw[jgal]-ycen3;
			                  CEz = zw[jgal]-zcen3;
			                  bot = 2*(wx*CEx+wy*CEy+wz*CEz-wx*ACx-wy*ACy-wz*ACz);
			                  x2 = top/bot;
			                  if ((x2>0)&&(x2<minx1)) {
				                minx1 = x2;
				                k4g1 = jgal;
			                  }
			                }
			                jgal=linklist[jgal];
			              }
			            }
		              }
		            }
		          }
		          //consider if this could be the right one
		          xcen41 = xcen3 + minx1*wx;
		          ycen41 = ycen3 + minx1*wy;
		          zcen41 = zcen3 + minx1*wz;
		          ux = xcen41 - xw[k1g];
		          uy = ycen41 - yw[k1g];
		          uz = zcen41 - zw[k1g];
		          vx = xcen41 - xw[k2g];
		          vy = ycen41 - yw[k2g];
		          vz = zcen41 - zw[k2g];
		          dotu1 = ux*wx + uy*wy + uz*wz;
		          dotv1 = vx*wx + vy*wy + vz*wz;
		          //swap u and v over - looking above and below the plane
		          modv = sqrt(pow(xw[k1g]-xw[k2g],2)+pow(yw[k1g]-yw[k2g],2)+pow(zw[k1g]-zw[k2g],2));
		          vx = (xw[k1g]-xw[k2g])/modv;
		          vy = (yw[k1g]-yw[k2g])/modv;
		          vz = (zw[k1g]-zw[k2g])/modv;
		          modu = sqrt(pow(xw[k3g]-xw[k2g],2)+pow(yw[k3g]-yw[k2g],2)+pow(zw[k3g]-zw[k2g],2));
		          ux = (xw[k3g]-xw[k2g])/modu;
		          uy = (yw[k3g]-yw[k2g])/modu;
		          uz = (zw[k3g]-zw[k2g])/modu;
		          //new vector is
		          wx = (uy*vz-uz*vy);
		          wy = (-ux*vz+uz*vx);
		          wz = (ux*vy-uy*vx);
		          modw = sqrt(pow(wx,2)+pow(wy,2)+pow(wz,2));
		          //now as before. Move on wx,wy,wz from
		          //already know which cells to offset from. reset minx though
		          minx2 = 100000.;
		          for (meshi=max(0,ixc-ioff);meshi<min(ngrid,ixc+ioff);meshi++) {
		            for (meshj=max(0,iyc-ioff);meshj<min(ngrid,iyc+ioff);meshj++) {
		              for (meshk=max(0,izc-ioff);meshk<min(ngrid,izc+ioff);meshk++) {
			            jgal = chainlist[meshi][meshj][meshk];
			            if (jgal!=-1) {
			              stopat = -1;
			              jgal = linklist[jgal];
			              while (jgal!=stopat) {
			                if (stopat==-1)
			                  stopat = jgal;
			                if ((jgal!=k1g)&&(jgal!=k2g)&&(jgal!=k3g)) {
			                  sep = sqrt(pow(xw[jgal]-xcen3,2)+pow(yw[jgal]-ycen3,2)+pow(zw[jgal]-zcen3,2));
			                  ACx = xw[k1g] - xcen3;
			                  ACy = yw[k1g] - ycen3;
			                  ACz = zw[k1g] - zcen3;
			                  top = pow(sep,2)-pow(rad,2);
			                  CEx = xw[jgal] - xcen3;
			                  CEy = yw[jgal] - ycen3;
			                  CEz = zw[jgal] - zcen3;
			                  bot = 2*(wx*CEx+wy*CEy+wz*CEz-wx*ACx-wy*ACy-wz*ACz);
			                  x2 = top/bot;
			                  if ((x2>0)&&(x2<minx2)) {
				                minx2 = x2;
				                k4g2 = jgal;
			                  }
			                }
			                jgal = linklist[jgal];
			              }
			            }
		              }
		            }
		          }

		          xcen42 = xcen3 + minx2*wx;
		          ycen42 = ycen3 + minx2*wy;
		          zcen42 = zcen3 + minx2*wz;
		          ux = xcen42 - xw[k1g];
		          uy = ycen42 - yw[k1g];
		          uz = zcen42 - zw[k1g];
		          vx = xcen42 - xw[k2g];
		          vy = ycen42 - yw[k2g];
		          vz = zcen42 - zw[k2g];
		          dotu2 = ux*wx + uy*wy + uz*wz;
		          dotv2 = vx*wx + vy*wy + vz*wz;
		          flg = 0;

		          // printf("Found 4th gal\n");

		          if ((minx1<1000)&&(dotu1>0)) {
		            flg = 1;
		            xcen4 = xcen41;
		            ycen4 = ycen41;
		            zcen4 = zcen41;
		            k4g = k4g1;
		          }
		          if ((minx2<1000)&&(dotu2>0)) {
		            flg = 2;
		            xcen4 = xcen42;
		            ycen4 = ycen42;
		            zcen4 = zcen42;
		            k4g = k4g2;
		          }
		          if ((dotu1>0)&&(dotu2>0)) {
		            if ((minx1<minx2)&&(minx1<1000)) {
		              flg = 3;
		              xcen4 = xcen41;
		              ycen4 = ycen41;
		              zcen4 = zcen41;
		              k4g = k4g1;
		            }
		            else if ((minx2<minx1)&&(minx2<1000)) {
		              flg = 4;
		              xcen4 = xcen42;
		              ycen4 = ycen42;
		              zcen4 = zcen42;
		              k4g = k4g2;
		            }
		          }

		          if (flg == 0)
		            continue;
		          r = sqrt(pow(xcen4,2)+pow(ycen4,2)+pow(zcen4,2));
		          ra = atan(ycen4/xcen4)*180./pi;
		          dec = asin(zcen4/r)*180./pi;
		          if ((ycen4>0)&&(xcen4<0)) 
		            ra += 180.;
		          if ((ycen4<0)&&(xcen4<0)) 
		            ra += 180.;
		          if (ra<0.)
		            ra += 360.;
		          /*
		          printf("%f %f %f\n",xcen,ycen,zcen);
		          printf("%f %f %f\n",xcen2,ycen2,zcen2);
		          printf("%f %f %f\n",xcen3,ycen3,zcen3);
		          printf("%f %f %f\n",xcen4,ycen4,zcen4);
		          */
		          if ((mask[int(ra)][int(dec)-decoffset]==0)||(r>maxdist)||(r<mindist))
		            continue;
		          if (flg > 0) {
		    
		            //these 4 numbers should all be the same
		            sep = sqrt(pow(xcen4-xw[k1g],2)+pow(ycen4-yw[k1g],2)+pow(zcen4-zw[k1g],2));
		            rad = sqrt(pow(xcen4-xw[k2g],2)+pow(ycen4-yw[k2g],2)+pow(zcen4-zw[k2g],2));
		            rad2 = sqrt(pow(xcen4-xw[k3g],2)+pow(ycen4-yw[k3g],2)+pow(zcen4-zw[k3g],2));
		            rad3 = sqrt(pow(xcen4-xw[k4g],2)+pow(ycen4-yw[k4g],2)+pow(zcen4-zw[k4g],2));
		            if ((xcen4>xmin)&&(xcen4<xmax)&&(ycen4>ymin)&&(ycen4<ymax)&&(zcen4>zmin)&&(zcen4<zmax)) {
		              ncheck = 0;
		              ncheck2 = 0;
		              for (p=0;p<1000;p++) {
			            //RANDOM NUMBER GEN
			            x = rand()/(RAND_MAX+1.0)*rad*2+xcen4-rad;
			            y = rand()/(RAND_MAX+1.0)*rad*2+ycen4-rad;
			            z = rand()/(RAND_MAX+1.0)*rad*2+zcen4-rad;

			            diff = pow(x-xcen4,2)+pow(y-ycen4,2)+pow(z-zcen4,2);
			            if (diff<pow(rad,2)) {
			              ncheck++;
			              if ((x>xmin)&&(x<xmax)&&(y>ymin)&&(y<ymax)&&(z>zmin)&&(z<zmax)) {
			                //test if actually in simulation
			                ncheck2++;
			              }
			            }
		              }
		              //		      printf("%d %d\n",ncheck,ncheck2);
		              if (ncheck2>0.95*ncheck) {
			            diff = rad3-rad;
			            if (sqrt(pow(diff,2))>0.1) {
			              printf("problem %f %d %d %d %f %f %d\n",diff,i,j,k,rad,rad3,flg);
			            }
			            xvd[nvd] = xcen4;
			            yvd[nvd] = ycen4;
			            zvd[nvd] = zcen4;
			            rvd[nvd] = rad;
			            myvoids[nvd][0] = xcen4;
			            myvoids[nvd][1] = ycen4;
			            myvoids[nvd][2] = zcen4;
			            myvoids[nvd][3] = rad;
			            nvd++;
			            //fillgrid(mygrid,xcen4,ycen4,zcen4,rad,dl);
		              }
		            }
		          }
		        }
	          }
	        }
	      }
	    }
      }
    }
  }
  printf("found a total of %d potential voids\n",nvd);
  printf("skipped a total of %d gridcells\n",skiptot);
  
  //need to sort the potential voids into size order
  //QSORT THIS
  qsort(myvoids,nvd);

  printf("sorted\n");

  for (i=0;i<nvd;i++) {
    xvd[i] = myvoids[i][0];
    yvd[i] = myvoids[i][1];
    zvd[i] = myvoids[i][2];
    rvd[i] = myvoids[i][3];
  }

  npv = nvd;

  //the largest hole is a void
  nv = 0;

  for (i=0;i<nvd;i++) {
    flag[i] = 0;
  }

  nh = 0;
  for (i=0;i<nvd;i++) {
    if (rvd[i]>10.) { //this is where we impose r>10
      ntouch = 0;
      nspecial = 0;
      flag[i] = nv;
      for (j=0;j<nv;j++) {
	    sep = sqrt(pow(xvd[i]-xv[j],2)+pow(yvd[i]-yv[j],2)+pow(zvd[i]-zv[j],2));
	    diff = rvd[i] + rv[j];
	    //calculate the overlapping volume.  If greater than half the volume
	    //of the smaller sphere, then accept it.  Otherwise don't.
	    r1 = rvd[i];
	    r2 = rv[j];
	    //check r2>r1
	    if (sep<1) {
	      ntouch = 1;
	      flag[i] = j;
	    }
	    else if (sep<=(r2+r1)) {
	      /*
	      a1 = (pow(sep,2)+pow(r1,2)-pow(r2,2))/2./sep;
	      a2 = sep-a1;
	      v = 2*pi*(pow(r1,3)+pow(r2,3))/3. - pi*(pow(r1,2)*a1+pow(r2,2)*a2-pow(a1,3)/3-pow(a2,3)/3);
	      */
	      v = pi/(12*sep)*pow(r1+r2-sep,2)*(pow(sep,2)+2*sep*r2-3*pow(r2,2)+2*sep*r1+6*r2*r1-3*pow(r1,2));
	      vs = 4.*pi*pow(r1,3)/3.;
	      if (v>(frac*vs)) {
	        //these two spheres overlap by more than X%, sphere is not unique
	        ntouch = 1;
	        flag[i] = j;
	      }
	    }
      }
      xh[nh] = xvd[i];
      yh[nh] = yvd[i];
      zh[nh] = zvd[i];
      rh[nh] = rvd[i];
      flog[nh] = flag[i];
      nh++;
      if (ntouch==0) {
	    xv[nv] = xvd[i];
	    yv[nv] = yvd[i];
	    zv[nv] = zvd[i];
	    rv[nv] = rvd[i];
	    nv++;
      }
    }
  }
  printf("ngroup is %d\n",nv);

  for (i=0;i<nv;i++) {
    n[i]=0;
  }

  for (i=0;i<nh;i++) {
    fprintf(out2,"%f %f %f %f %d\n",xh[i],yh[i],zh[i],rh[i],flog[i]);
    n[flog[i]]++;
  }

  ngroup = nv;
  nran = 10000;
  for (i=0;i<ngroup;i++) {
    nsph = 0;
    rad = rv[i]*4;
    for (j=0;j<nran;j++) {
      x = rand()/(RAND_MAX+1.0)*rad + xv[i] - rad/2.;
      y = rand()/(RAND_MAX+1.0)*rad + yv[i] - rad/2.;
      z = rand()/(RAND_MAX+1.0)*rad + zv[i] - rad/2.;
      test = 0;
      for (k=0;k<nh;k++) {
	    if (flog[k]==i) {
	      //calculate difference between particle and sphere
	      sep = pow(x-xh[k],2)+pow(y-yh[k],2)+pow(z-zh[k],2);
	      if (sep<pow(rh[k],2)) {
	        //this particle lies in at least one sphere
	        nsph++;
	        break;
	      }
	    }
      }
    }
    voidvol[i] = pow(rad,3)*nsph/(float)nran;
  }
  
  for (i=0;i<nh;i++) {
    nfield[i] = 0;
  }

  for (i=0;i<nf;i++) {
    for (j=0;j<nh;j++) {
      if (dist(xf[i],yf[i],zf[i],xh[j],yh[j],zh[j])<rh[j]) {
	    nfield[flog[j]]++;
	    fprintf(voidgals,"%f %f %f %d\n",xf[i],yf[i],zf[i],flog[j]);
	    break;
      }
    }
  }

  nin = nwall + nf;
  
  for (i=0;i<ngroup;i++) {
    deltap = (nfield[i]-nin*voidvol[i]/vol)/(nin*voidvol[i]/vol);
    sep = pow(voidvol[i]*3/4/pi,1.0/3.0);
    r = sqrt(pow(xv[i],2)+pow(yv[i],2)+pow(zv[i],2));
    decv = asin(zv[i]/r)*180/pi;
    rav = atan(yv[i]/xv[i])*180/pi;
    if ((yv[i]>0)&&(xv[i]<0))
      rav+=180;
    if ((yv[i]<0)&&(xv[i]<0))
      rav+=180;
    rad = 4*pi*pow(rv[i],3)/3.0/voidvol[i];
    fprintf(out3,"%f %f %f %f %f %f %f %d %f\n",rv[i],sep,voidvol[i],xv[i],yv[i],zv[i],deltap,nfield[i],rad);
    fprintf(out1,"%f %f %f %f %f %f %f\n",xv[i],yv[i],zv[i],rv[i],r,rav,decv);
  }

  //go back through galaxy list and find void galaxies



}
