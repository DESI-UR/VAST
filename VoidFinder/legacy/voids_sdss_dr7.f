

c a huge program to do EVERYTHING that I want to do with voids in SDSS

      implicit none  
      integer ntot, nv, ngb
      integer NGALMAX, ngrid, CHAINMAX, CHAINMAXX, CHAINMAXY, CHAINMAXZ
      parameter(ntot=500000)
      parameter (NGALMAX=500000,NGRID=128,CHAINMAX=100)
      parameter (CHAINMAXX=200,CHAINMAXY=250,CHAINMAXZ=250)

      integer ne1, ne2, nl1, nl2, decoffset
      integer inout, nintest, inout2
      real pixsize, lamc, etac, dr7
      parameter (ne1 = 1602,ne2=800,nl1=5064,nl2=4576) ! DR4plus
      parameter (pixsize=0.025)
      real dec(ne1), ra(ne1)
      real dtor, rtod

      integer  nin, test
      integer chainlist(CHAINMAXX,CHAINMAXY,CHAINMAXZ)
      integer linklist(NGALMAX), nwall, nf
      integer tempchain(CHAINMAXX,CHAINMAXY,CHAINMAXZ)
      integer meshi,meshj,meshk, ngal(CHAINMAXX,CHAINMAXY,CHAINMAXZ)
      integer stopat, jgal, igx, igy, igz
      integer i, j, k, k1g, nposs, ioff, igal
      integer ncell, k2g, k3g, ixc, iyc, izc, k4g
      integer ncheck, ncheck2, flg, k4g1, k4g2
      integer flag(NGALMAX), flog(NGALMAX)
      integer nfield(NGALMAX), nsph, nran, flagg(NGALMAX)
      integer NGROUPMAX, ntouch
      parameter (NGROUPMAX=100000)
      integer ngroup, nspecial, n(NGROUPMAX)
      integer npv, nh, npvh, dosep, mi, mj,mk
      integer mask(360,180)

      real l, sd, var, avsep, minsep2(NGALMAX)
      real totsep, minsep3(NGALMAX),minsep1(NGALMAX)
      real c, zmax, rac, decc
      real a1, a2, v, vs, r1, r2, r3
      real dmax, raf, decf, dmin
      real wksp(NGALMAX), iwksp(NGALMAX)
      real xin(NGALMAX), yin(NGALMAX), zin(NGALMAX)
      real xw(NGALMAX), yw(NGALMAX), zw(NGALMAX)
      real xv(NGALMAX), yv(NGALMAX), zv(NGALMAX), rv(NGALMAX)
      real xf(NGALMAX), yf(NGALMAX), zf(NGALMAX)
      real xvd(NGALMAX), yvd(NGALMAX), zvd(NGALMAX)
      real rvd(NGALMAX), decvd(NGALMAX), ravd(NGALMAX), dvd(NGALMAX)
      real xpv(NGALMAX), ypv(NGALMAX), zpv(NGALMAX)
      real rh(NGALMAX)
      real xh(NGALMAX), yh(NGALMAX), zh(NGALMAX)
      real rpv(NGALMAX), decpv(NGALMAX), rapv(NGALMAX), dpv(NGALMAX)
      real vol, d1, box, minsep, near, zdiff, ydiff, xdiff
      real xcen, ycen, zcen, rad, sep, zcen2, ycen2, xcen2   
      real xmin,xmax,ymin,ymax,zmin, minx, modv, voidvol(NGALMAX)
      real x2, top, bot, BAx, BAy, BAz, vx, vy, vz
      real rad2, xcen3, ycen3, zcen3, xbi, ybi, zbi
      real CEx, CEy, CEz, ACx, ACy, ACz
      real rad3, dotu2, dotv2, xcen4, ycen4, zcen4
      real wx, wy, wz, modw, ux, uy, uz, modu, rmin
      real pi, ran3, seed, diff, dec4, ra4, ra2, ra3, dec2, dec3
      real minx1, minx2, r4, frac
      real xcen41, ycen41, zcen41, xcen42, ycen42, zcen42
      real dotu1, dotv1, x, y, z, rav, decv, deltap, r
      real omega0, lambda0, beta, zplus1, uri, rai, deci, dlum
      real dlzcomb, lambda, eta,zi,rabsmagi

      character*80 out1, out2, out3, infile

c      write(0,*) 'enter name of files'
c      read(*,'(A)') infile
c      read(*,'(A)') out1
c      read(*,'(A)') out2
c      read(*,'(A)') out3
c      open(1,file=infile,status='unknown')
c      open(12,file=out1,status='unknown')
c      open(72,file=out2,status='unknown')
c      open(9,file=out3,status='unknown')

c      write(*,*) 'enter the value of frac'
c      read(*,*) frac
      frac = 0.1
      write(*,*) 'using this fraction', frac
      write(*,*) 'enter the value of dosep'
c      read(*,*) dosep
      dosep = 1
      write(*,*) 'using this', dosep
      
c      dosep = 1 ! 0 to not search, 1 to search

      open(1,file='SDSSdr7/vollim_dr7_cbp_102709.dat',status='unknown')
      open(12,file='o1',status='unknown')
      open(72,file='o2',status='unknown')
      open(9,file='o3',status='unknown')

c constants used
      c = 3.0e+5
      pi = 4.*atan(1.)
      dtor = 0.0174532925
      rtod = 57.29577951
      decoffset = -90

c specify things

      dmax = 300.   ! This depends of VL samples in use (312 is the vol lim so this means that within one radius or so it is in the survey)
      dmin = 0.

      do i = 1, 360
         do j = 1, 180
            mask(i,j) = 0
         end do
      end do

c read mask
      write(*,*) 'reading mask'
      open(7,file='SDSSdr7/cbpdr7mask.dat',status='unknown')
      do i=1,ntot
         read(7,*,end=15) x, y
c         write(*,*) x, y, int(x)+1, int(y)-decoffset, i
         mask(int(x)+1,int(y)-decoffset) = 1
      end do
      write(*,*) 'mask read'

c initialisations
 15   npv = 0
      seed = -78197341
      box = 630.    ! We'll have to see how big we need
      d1 = box/float(ngrid)

      write(*,*) 'number of grid cells is ', ngrid, d1, box
      ncell = 0 
c read in the volume limited SDSS sample

      nin = 0
      xmin = 1000
      ymin = 1000
      zmin = 1000

      xmax = -1000
      ymax = -1000
      zmax = -1000


      do i = 1, ntot
         read(1,*,end=98) raf, decf, z, r
         xin(i) = r*cos(raf*dtor)*cos(decf*dtor)
         yin(i) = r*sin(raf*dtor)*cos(decf*dtor)
         zin(i) = r*sin(decf*dtor)
         if(i.EQ.1) write(*,*) xin(i), yin(i), zin(i)
         nin = nin + 1
         xmin = min(xin(i),xmin)
         ymin = min(yin(i),ymin)
         zmin = min(zin(i),zmin)
         xmax = max(xin(i),xmax)
         ymax = max(yin(i),ymax)
         zmax = max(zin(i),zmax)
      end do
 98   write(*,*) 'there are', nin, '  galaxies in this SDSS sample'

      write(*,*) xmin, xmax
      write(*,*) ymin, ymax
      write(*,*) zmin, zmax


      do meshi=1,ngrid
         do meshj=1,ngrid
            do meshk=1,ngrid
               chainlist(meshi,meshj,meshk)=-1
               tempchain(meshi,meshj,meshk) = 0
            enddo
         enddo
      enddo

      write(*,*) 'making the grid'
      do igal=1,nin
         meshi = (xin(igal)-xmin)/d1 + 1
         meshj = (yin(igal)-ymin)/d1 + 1
         meshk = (zin(igal)-zmin)/d1 + 1
         if (chainlist(meshi,meshj,meshk).eq.-1) then
            chainlist(meshi,meshj,meshk)=igal
            tempchain(meshi,meshj,meshk)=igal
         else
            linklist(tempchain(meshi,meshj,meshk))=igal
            tempchain(meshi,meshj,meshk)=igal
         endif
      enddo

      write(*,*) 'made the grid'
      do meshi=1,ngrid
         do meshj=1,ngrid
            do meshk=1,ngrid
               if (chainlist(meshi,meshj,meshk).ne.-1) 
     &  linklist(tempchain(meshi,meshj,meshk))=
     &  chainlist(meshi,meshj,meshk)
            enddo
         enddo
      enddo
      write(*,*) 'made the grid'

      do i = 1, ngrid
         do j = 1, ngrid
            do k = 1, ngrid
               ngal(i,j,k) = 0
            end do
         end do
      end do    

      do i = 1, nin
         igx = (xin(i)-xmin)/d1 + 1
         igy = (yin(i)-ymin)/d1 + 1
         igz = (zin(i)-zmin)/d1 + 1
         ngal(igx,igy,igz) = ngal(igx,igy,igz)+1
      end do

      nintest = nin

c      if(dosep.EQ.1) then ! still need to find minsep3 values whether calculating l or not so have to use this loop
         write(*,*) 'finding sep'
         totsep = 0.
         do i = 1, nintest
            minsep1(i) = 10000000.
            minsep2(i) = 10000000.
            minsep3(i) = 10000000.
            if(mod(i,10000).EQ.0) write(*,*) i
            mi = (xin(i)-xmin)/d1 + 1
            mj = (yin(i)-ymin)/d1 + 1
            mk = (zin(i)-zmin)/d1 + 1
            ioff = 0
            nposs = 0
            do while (nposs.LT.6)
               ioff = ioff + 1
               do meshi=max(1,mi-ioff),min(ngrid,mi+ioff)
                  do meshj=max(1,mj-ioff),min(ngrid,mj+ioff)
                     do meshk=max(1,mk-ioff),min(ngrid,mk+ioff)
                        nposs = nposs + ngal(meshi,meshj,meshk)
                     end do
                  end do
               end do
            end do
c     now need to check that isn't one closer in shell just round it.
            ioff = ioff + 1
            do meshi=max(1,mi-ioff),min(ngrid,mi+ioff)
               do meshj=max(1,mj-ioff),min(ngrid,mj+ioff)
                  do meshk=max(1,mk-ioff),min(ngrid,mk+ioff)
                     jgal=chainlist(meshi,meshj,meshk)
                     if (jgal.ne.-1) then
                        stopat=-1
                        jgal=linklist(jgal)
                        do while (jgal.ne.stopat)
                           if (stopat.eq.-1) stopat=jgal
                           sep=((xin(i)-xin(jgal))**2. +
     &                  (yin(i)-yin(jgal))**2. + (zin(i)-zin(jgal))**2.)
                           if(sep.GT.0) then
                              if(sep.LT.minsep1(i)) then
                                 minsep3(i) = minsep2(i)
                                 minsep2(i) = minsep1(i)
                                 minsep1(i) = sep
                  else if (sep.GE.minsep1(i).AND.sep.LE.minsep2(i)) then
                                 minsep3(i) = minsep2(i)
                                 minsep2(i) = sep
                  else if (sep.GE.minsep2(i).AND.sep.LE.minsep3(i)) then
                                 minsep3(i) = sep
                              end if
                           end if
                           jgal=linklist(jgal)
                        end do                               
                     end if
                  end do
               end do
            end do
            minsep3(i) = sqrt(minsep3(i))
            minsep2(i) = sqrt(minsep2(i))
            minsep1(i) = sqrt(minsep1(i))
            totsep = minsep3(i) + totsep
c            write(*,*) minsep1(i), minsep2(i), minsep3(i), i
            if(minsep3(i).GT.20) write(*,*) minsep3(i), i
         end do
         avsep = totsep/float(nintest)
         write(*,*) 'average seperation to n3rd gal is', avsep
         var = 0.
         sd = 0.
         do i = 1, nintest
            var = (minsep3(i) - avsep)**2. + var
         end do
         sd = sqrt(var/float(nintest))
         write(*,*) 'the standard deviation is', sd
         
c     now know the search radius
         l = avsep + 1.5*sd
c     search again to flag gals with more than 3 neighbours in l as
c     wall galaxies.

      if (dosep.EQ.0) then
         l = 7.0
      end if
      
      write(*,*) 'going to build wall with search value', l



      nf= 0
      nwall = 0
      do i = 1, nin
         if(minsep3(i).GT.l) then
c these are the field galaxies - potential void galaxies 
          nf = nf + 1
          xf(nf) = xin(i)
          yf(nf) = yin(i)
          zf(nf) = zin(i)
          write(2,*) xf(nf), yf(nf), zf(nf)
        else if(minsep3(i).LE.l) then
          nwall = nwall + 1
          xw(nwall) = xin(i)
          yw(nwall) = yin(i)
          zw(nwall) = zin(i)
          write(3,*) xw(nwall), yw(nwall), zw(nwall)
       end if
      end do
      write(*,*) nf, nwall, 'I am here'

      xmin = 1000
      ymin = 1000
      zmin = 1000
      xmax = -1000
      ymax = -1000
      zmax = -1000

      do i = 1, nwall
         xmin = min(xw(i),xmin)
         ymin = min(yw(i),ymin)
         zmin = min(zw(i),zmin)
         xmax = max(xw(i),xmax)
         ymax = max(yw(i),ymax)
         zmax = max(zw(i),zmax)         
      end do
      write(*,*) xmax, ymax, zmax
      write(*,*) xmin, ymin, zmin
      write(*,*) xmax - xmin, ymax-ymin, zmax-zmin

c     Sort the galaxies into a chaining mesh.
      write (0,*) 'Constructing galaxy chaining mesh'

      do meshi=1,ngrid
         do meshj=1,ngrid
            do meshk=1,ngrid
               chainlist(meshi,meshj,meshk)=-1
               tempchain(meshi,meshj,meshk) = 0
            enddo
         enddo
      enddo

      do igal=1,nwall
         meshi = (xw(igal)-xmin)/d1 + 1
         meshj = (yw(igal)-ymin)/d1 + 1
         meshk = (zw(igal)-zmin)/d1 + 1
         if (chainlist(meshi,meshj,meshk).eq.-1) then
            chainlist(meshi,meshj,meshk)=igal
            tempchain(meshi,meshj,meshk)=igal
         else
            linklist(tempchain(meshi,meshj,meshk))=igal
            tempchain(meshi,meshj,meshk)=igal
         endif
      enddo

      do meshi=1,ngrid
         do meshj=1,ngrid
            do meshk=1,ngrid
               if (chainlist(meshi,meshj,meshk).ne.-1) 
     &  linklist(tempchain(meshi,meshj,meshk))=
     &  chainlist(meshi,meshj,meshk)
            enddo
         enddo
      enddo

c first find an empty cell

      do i = 1, ngrid
         do j = 1, ngrid
            do k = 1, ngrid
               ngal(i,j,k) = 0
            end do
         end do
      end do    

      do i = 1, nwall
         igx = (xw(i)-xmin)/d1 + 1
         igy = (yw(i)-ymin)/d1 + 1
         igz = (zw(i)-zmin)/d1 + 1
         ngal(igx,igy,igz) = ngal(igx,igy,igz)+1
      end do

      nh = 0
      do i = 1, ngrid
c      do i = 25, 30
         write(*,*) i
         do j = 1, ngrid
c         do j = 7, 12
            do k = 1, ngrid
c            do k = 5, 10
               xcen = (i-0.5)*d1 + xmin 
               ycen = (j-0.5)*d1 + ymin 
               zcen = (k-0.5)*d1 + zmin 
c               if(xcen.GT.-70.AND.xcen.LT.-40.AND.ycen.GT.-200.AND.
c     &    ycen.LT.-170.AND.zcen.GT.40.AND.zcen.LT.70) write(*,*) i, j, k

               r = sqrt(xcen**2. + ycen**2. + zcen**2.)
               if(ngal(i,j,k).EQ.0.AND.r.GT.dmin
     &                   .AND.r.LT.dmax.AND.xcen.LT.0) then
c xcen LT 0 must be a constraint from the galaxy geometry
                  rac = atan(ycen/xcen)*180./pi
                  decc = asin(zcen/r)*180./pi
                  if(ycen.GT.0.AND.xcen.LT.0) rac = rac + 180.
                  if(ycen.LT.0.AND.xcen.LT.0) rac = rac + 180.
                  call checkdr7(rac,decc,inout2)
                  inout = mask(int(rac)+1,int(decc)-decoffset)
                  if (inout.EQ.1) then
c     this is an empty cell and worth growing a void from it
                     ioff = 0
                     nposs = 0
                     do while (nposs.LT.1)
                        ioff = ioff + 1
                        do meshi=max(1,i-ioff),min(ngrid,i+ioff)
                           do meshj=max(1,j-ioff),min(ngrid,j+ioff)
                              do meshk=max(1,k-ioff),min(ngrid,k+ioff)
                                 nposs = nposs + ngal(meshi,meshj,meshk)
                              end do
                           end do
                        end do
                     end do
c     now need to check that isn't one closer in shell just round it.
                     ioff = ioff + 1
                     minsep = 100000.
                     do meshi=max(1,i-ioff),min(ngrid,i+ioff)
                        do meshj=max(1,j-ioff),min(ngrid,j+ioff)
                           do meshk=max(1,k-ioff),min(ngrid,k+ioff)
                              jgal=chainlist(meshi,meshj,meshk)
                              if (jgal.ne.-1) then
                                 stopat=-1
                                 jgal=linklist(jgal)
                                 do while (jgal.ne.stopat)
                                    if (stopat.eq.-1) stopat=jgal
                                    near=sqrt((xcen-xw(jgal))**2.+
     &                        (ycen-yw(jgal))**2. + (zcen-zw(jgal))**2.)
                                    if(near.LT.minsep) then
                                       minsep = near
                                       k1g = jgal
                                    end if
                                    jgal=linklist(jgal)
                                 end do                               
                              end if
                           end do
                        end do
                     end do

c     now found the first galaxy that is close to the center of the empty cell
                    modv = sqrt((xcen-xw(k1g))**2. + (ycen-yw(k1g))**2 + 
     &                    (zcen-zw(k1g))**2)
                     vx = (xw(k1g)-xcen)/modv
                     vy = (yw(k1g)-ycen)/modv
                     vz = (zw(k1g)-zcen)/modv
c     Andrew's way at getting the next galaxy works.
                     minx = 10000
                     k2g = 0
c     want to find the next closest galaxy. Minimise x = top/bottom
c     again need to find ioff and look in ioff + 1 cells to find this
c     still based in first cell here
c     however, next nearest galaxy call lie ANYWHERE on vector that
c     joins Xcen1 and Gal1. 
                     ioff = ngrid
                     minsep = 100000.
                     do meshi=max(1,i-ioff),min(ngrid,i+ioff)
                        xdiff=(meshi-0.5)*d1 + xmin
                        do meshj=max(1,j-ioff),min(ngrid,j+ioff)
                           ydiff=(meshj-0.5)*d1 + ymin
                           do meshk=max(1,k-ioff),min(ngrid,k+ioff)
                              zdiff=(meshk-0.5)*d1 + zmin
                              jgal=chainlist(meshi,meshj,meshk)
                              if (jgal.ne.-1) then
                                 stopat=-1
                                 jgal=linklist(jgal)
                                 do while (jgal.ne.stopat)
                                    if (stopat.eq.-1) stopat=jgal
                                    if(jgal.NE.k1g) then
                                       BAx = xw(k1g) - xw(jgal)
                                       BAy = yw(k1g) - yw(jgal)
                                       BAz = zw(k1g) - zw(jgal)
                                      bot = 2*(BAx*vx + BAy*vy + BAz*vz)
                                     top = (BAx**2. + BAy**2. + BAz**2.)
                                       x2 = top/bot
                                       if(x2.GT.0.AND.x2.LT.minx) then
                                          minx = x2
                                          k2g = jgal
                                       end if
                                    end if
                                    jgal=linklist(jgal)                      
                                 end do                               
                              end if
                           end do
                        end do
                     end do
                     x2 = minx
                     if(minx.LT.10000.) then
c     this is a problem - not possible to constrain void, 
c     probably at edge of survey
c     do not consider this center further
                        xcen2 = xw(k1g) - x2*vx
                        ycen2 = yw(k1g) - x2*vy
                        zcen2 = zw(k1g) - x2*vz
                     sep = sqrt((xcen2-xw(k1g))**2.+(ycen2-yw(k1g))**2.+
     &                       (zcen2-zw(k1g))**2.) 
                     rad = sqrt((xcen2-xw(k2g))**2.+(ycen2-yw(k2g))**2.+
     &                       (zcen2-zw(k2g))**2.) 
                        r2 = sqrt(xcen2**2. + ycen2**2. + zcen2**2.)
                        
                        dec2 = asin(zcen2/r2)*180./pi
                        ra2 = atan(ycen2/xcen2)*180./pi
                        
c                        write(*,*) xcen2, ycen2, zcen2, rad, sep
c     check conversion to ra and dec with various parts
                        if(ycen2.GT.0.AND.xcen2.LT.0) ra2 = ra2 + 180.
                        if(ycen2.LT.0.AND.xcen2.LT.0) ra2 = ra2 + 180.
c     check this is still in the survey
                        inout = mask(int(ra2)+1,int(dec2)-decoffset)
c                        call checkdr7(ra2,dec2,inout)
                        if(inout.EQ.1) then
c     now found second galaxy, have to find the third. Now need to know which
c     cell the new centre is in though.
                           ixc = (xcen2-xmin)/d1 + 1
                           iyc = (ycen2-ymin)/d1 + 1
                           izc = (zcen2-zmin)/d1 + 1
c     these are the bisection parts
                           xbi = (xw(k1g) + xw(k2g))/2.
                           ybi = (yw(k1g) + yw(k2g))/2.
                           zbi = (zw(k1g) + zw(k2g))/2.
c     need to find the bit I need to add on to (xcen2,ycen2,zcen2)
c     work out unit vector
                      modv = sqrt((xcen2 - xbi)**2. + (ycen2 - ybi)**2 + 
     &                          (zcen2 - zbi)**2)
                           vx = (xcen2-xbi)/modv
                           vy = (ycen2-ybi)/modv
                           vz = (zcen2-zbi)/modv
                           minx = 100000
c     very similar piece of code to that above 
c     but center box is where new center is
c     now need to check that isn't one closer in 2 shells just round it.
                           ioff = ngrid
                           do meshi=max(1,ixc-ioff),min(ngrid,ixc+ioff)
                              xdiff=(meshi-0.5)*d1 + xmin
                            do meshj=max(1,iyc-ioff),min(ngrid,iyc+ioff)
                                 ydiff=(meshj-0.5)*d1 + ymin
                            do meshk=max(1,izc-ioff),min(ngrid,izc+ioff)
                                    zdiff=(meshk-0.5)*d1 + zmin
                                    jgal=chainlist(meshi,meshj,meshk)
                                    if (jgal.ne.-1) then
                                       stopat=-1
                                       jgal=linklist(jgal)
                                       do while (jgal.ne.stopat)
                                          if (stopat.eq.-1) stopat=jgal
                                    if(jgal.NE.k1g.AND.jgal.NE.k2g) then
                                        sep = sqrt((xw(jgal)-xcen2)**2.+
     &                        (yw(jgal)-ycen2)**2.+(zw(jgal)-zcen2)**2.)
                                             ACx = xw(k1g) - xcen2
                                             ACy = yw(k1g) - ycen2
                                             ACz = zw(k1g) - zcen2
                                             top = sep**2. - rad**2. 
                                             CEx = xw(jgal) - xcen2
                                             CEy = yw(jgal) - ycen2 
                                             CEz = zw(jgal) - zcen2 
                     bot = 2*(vx*CEx+vy*CEy+vz*CEz-vx*ACx-vy*ACy-vz*ACz)
                                             x2 = top/bot
                                         if(x2.GT.0.AND.x2.LT.minx) then
                                                minx = x2
                                                k3g = jgal
                                             end if
                                          end if
                                          jgal=linklist(jgal)          
                                       end do                               
                                    end if
                                 end do
                              end do
                           end do
                           if(minx.LT.10000) then
                              xcen3 = xcen2 + minx*vx
                              ycen3 = ycen2 + minx*vy
                              zcen3 = zcen2 + minx*vz
                     sep = sqrt((xcen3-xw(k1g))**2.+(ycen3-yw(k1g))**2.+
     &                             (zcen3-zw(k1g))**2.) 
                     rad = sqrt((xcen3-xw(k2g))**2.+(ycen3-yw(k2g))**2.+
     &                             (zcen3-zw(k2g))**2.) 
                    rad2 = sqrt((xcen3-xw(k3g))**2.+(ycen3-yw(k3g))**2.+
     &                             (zcen3-zw(k3g))**2.) 
                              
                            r3 = sqrt(xcen3**2. + ycen3**2. + zcen3**2.)
                              dec3 = asin(zcen3/r3)*180./pi
                              ra3 = atan(ycen3/xcen3)*180./pi
c                          write(*,*) xcen3, ycen3, zcen3, sep, rad, rad2
                             
c     check conversion to ra and dec with various parts
                          if(ycen3.GT.0.AND.xcen3.LT.0) ra3 = ra3 + 180.
                          if(ycen3.LT.0.AND.xcen3.LT.0) ra3 = ra3 + 180.
c     check this is still in the survey
                          inout = mask(int(ra3)+1,int(dec3)-decoffset)
c                              call checkdr7(ra3,dec3,inout)
                              if(inout.EQ.1) then
c     and finally find the fourth galaxy - bit more involved but same idea.
                                 ixc = (xcen3-xmin)/d1 + 1
                                 iyc = (ycen3-ymin)/d1 + 1
                                 izc = (zcen3-zmin)/d1 + 1
                                 
c     this is the first run through                        
                  modu =sqrt((xw(k1g)-xw(k2g))**2.+(yw(k1g)-yw(k2g))**2.
     &                                + (zw(k1g)-zw(k2g))**2.)
                  modv=sqrt((xw(k2g)-xw(k3g))**2.+(yw(k2g)-yw(k3g))**2.+
     &                                (zw(k2g)-zw(k3g))**2.)
                                 ux = (xw(k1g)-xw(k2g))/modv
                                 uy = (yw(k1g)-yw(k2g))/modv
                                 uz = (zw(k1g)-zw(k2g))/modv
                                 vx = (xw(k3g)-xw(k2g))/modu
                                 vy = (yw(k3g)-yw(k2g))/modu
                                 vz = (zw(k3g)-zw(k2g))/modu
c     new vector is
                                 wx = (uy*vz-uz*vy)
                                 wy = (-ux*vz+uz*vx) ! second one has -'ve sign
                                 wz = (ux*vy-uy*vx)
                                 modw = sqrt(wx**2. +wy**2.+wz**2.)
                                 wx = wx/modw
                                 wy = wy/modw
                                 wz = wz/modw
                                 modw = sqrt(wx**2. +wy**2.+wz**2.)
c     now as before. Move on wx, wy, wz from 
                                 minx1 = 100000.
c     very similar piece of code to that above but center box 
c     is where new center is
c     now need to check that isn't one closer in 2 shells just round it.
                                 ioff = ngrid
                            do meshi=max(1,ixc-ioff),min(ngrid,ixc+ioff)
                                    xdiff=(meshi-0.5)*d1 + xmin
                            do meshj=max(1,iyc-ioff),min(ngrid,iyc+ioff)
                                       ydiff=(meshj-0.5)*d1 + ymin
                            do meshk=max(1,izc-ioff),min(ngrid,izc+ioff)
                                          zdiff=(meshk-0.5)*d1 + zmin
                                       jgal=chainlist(meshi,meshj,meshk)
                                          if (jgal.ne.-1) then
                                             stopat=-1
                                             jgal=linklist(jgal)
                                             do while (jgal.ne.stopat)
                                           if (stopat.eq.-1) stopat=jgal
                                          if(jgal.NE.k1g.AND.jgal.NE.k2g
     &                                            .AND.jgal.NE.k3g) then
                                        sep = sqrt((xw(jgal)-xcen3)**2.+
     &                        (yw(jgal)-ycen3)**2.+(zw(jgal)-zcen3)**2.)
                                                   ACx = xw(k1g) - xcen3
                                                   ACy = yw(k1g) - ycen3
                                                   ACz = zw(k1g) - zcen3
                                                 top = sep**2. - rad**2. 
                                                  CEx = xw(jgal) - xcen3
                                                  CEy = yw(jgal) - ycen3 
                                                  CEz = zw(jgal) - zcen3 
                     bot = 2*(wx*CEx+wy*CEy+wz*CEz-wx*ACx-wy*ACy-wz*ACz)
                                                   x2 = top/bot
                                        if(x2.GT.0.AND.x2.LT.minx1) then
                                                      minx1 = x2
                                                      k4g1 = jgal
                                                   end if
                                                end if
                                                jgal=linklist(jgal)    
                                             end do                               
                                          end if
                                       end do
                                    end do
                                 end do
c     consider if this could be the right one
                                 xcen41 = xcen3 + minx1*wx
                                 ycen41 = ycen3 + minx1*wy
                                 zcen41 = zcen3 + minx1*wz
                                 ux = xcen41 - xw(k1g)
                                 uy = ycen41 - yw(k1g)
                                 uz = zcen41 - zw(k1g)
                                 vx = xcen41 - xw(k2g)
                                 vy = ycen41 - yw(k2g)
                                 vz = zcen41 - zw(k2g)
                                 dotu1 = ux*wx + uy*wy + uz*wz
                                 dotv1 = vx*wx + vy*wy + vz*wz
                                 
c     if dotu and dotv positive then right way round, 
c     otherwise swap u and v over
c     however if the are negative have to consider others
                                 
                 modv = sqrt((xw(k1g)-xw(k2g))**2.+(yw(k1g)-yw(k2g))**2.
     &                                + (zw(k1g)-zw(k2g))**2.)
                                 vx = (xw(k1g)-xw(k2g))/modv
                                 vy = (yw(k1g)-yw(k2g))/modv
                                 vz = (zw(k1g)-zw(k2g))/modv
                 modu = sqrt((xw(k2g)-xw(k3g))**2.+(yw(k2g)-yw(k3g))**2.
     &                                + (zw(k2g)-zw(k3g))**2.)
                                 ux = (xw(k3g)-xw(k2g))/modu
                                 uy = (yw(k3g)-yw(k2g))/modu
                                 uz = (zw(k3g)-zw(k2g))/modu
                                 
c     new vector is
                                 wx = (uy*vz-uz*vy)
                                 wy = (-ux*vz+uz*vx) ! second one has -'ve sign
                                 wz = (ux*vy-uy*vx)
                                 modw = sqrt(wx**2. +wy**2.+wz**2.)
                                 wx = wx/modw
                                 wy = wy/modw
                                 wz = wz/modw
                                 modw = sqrt(wx**2. +wy**2.+wz**2.)
c     now as before. Move on wx, wy, wz from
c     already know which cells to offset from. reset minx though
                                 minx2 = 100000.
                            do meshi=max(1,ixc-ioff),min(ngrid,ixc+ioff)
                            do meshj=max(1,iyc-ioff),min(ngrid,iyc+ioff)
                            do meshk=max(1,izc-ioff),min(ngrid,izc+ioff)
                                       jgal=chainlist(meshi,meshj,meshk)
                                          if (jgal.ne.-1) then
                                             stopat=-1
                                             jgal=linklist(jgal)
                                             do while (jgal.ne.stopat)
                                           if (stopat.eq.-1) stopat=jgal
                                          if(jgal.NE.k1g.AND.jgal.NE.k2g
     &                                            .AND.jgal.NE.k3g) then
                                        sep = sqrt((xw(jgal)-xcen3)**2.+
     &                        (yw(jgal)-ycen3)**2.+(zw(jgal)-zcen3)**2.)
                                                   ACx = xw(k1g) - xcen3
                                                   ACy = yw(k1g) - ycen3
                                                   ACz = zw(k1g) - zcen3
                                                 top = sep**2. - rad**2. 
                                                  CEx = xw(jgal) - xcen3
                                                  CEy = yw(jgal) - ycen3 
                                                  CEz = zw(jgal) - zcen3
                     bot = 2*(wx*CEx+wy*CEy+wz*CEz-wx*ACx-wy*ACy-wz*ACz)
                                                   x2 = top/bot
                                       if(x2.GT.0.AND.x2.LT.minx2) then
                                                      minx2 = x2
                                                      k4g2 = jgal
                                                   end if
                                                end if
                                                jgal=linklist(jgal)          
                                             end do                
                                          end if
                                       end do
                                    end do
                                 end do
                                 xcen42 = xcen3 + minx2*wx
                                 ycen42 = ycen3 + minx2*wy
                                 zcen42 = zcen3 + minx2*wz
                                 ux = xcen42 - xw(k1g)
                                 uy = ycen42 - yw(k1g)
                                 uz = zcen42 - zw(k1g)
                                 vx = xcen42 - xw(k2g)
                                 vy = ycen42 - yw(k2g)
                                 vz = zcen42 - zw(k2g)
                                 dotu2 = ux*wx + uy*wy + uz*wz
                                 dotv2 = vx*wx + vy*wy + vz*wz
                                 flg = 0
                                 
c     write(*,*) minx1, minx2
                                 
                                 if(minx1.LT.1000.AND.dotu1.GT.0) then
                                    flg = 1
                                    xcen4 = xcen41
                                    ycen4 = ycen41
                                    zcen4 = zcen41
                                    k4g = k4g1
                                 end if
                                 if (minx2.LT.1000.AND.dotu2.GT.0) then
                                    flg = 1
                                    xcen4 = xcen42
                                    ycen4 = ycen42
                                    zcen4 = zcen42
                                    k4g = k4g2
                                 end if
                                 if (dotu1.GT.0.AND.dotu2.GT.0) then
                              if(minx1.LT.minx2.AND.minx1.LT.1000.) then
                                       flg = 1
                                       xcen4 = xcen41
                                       ycen4 = ycen41
                                       zcen4 = zcen41
                                       k4g = k4g1
                        else if (minx2.LT.minx1.AND.minx2.LT.1000.) then
                                       flg = 1
                                       xcen4 = xcen42
                                       ycen4 = ycen42
                                       zcen4 = zcen42
                                       k4g = k4g2
                                    end if
                                 end if
                                 if(flg.GT.0) then
                     sep = sqrt((xcen4-xw(k1g))**2.+(ycen4-yw(k1g))**2.+
     &                                   (zcen4-zw(k1g))**2.) 
                     rad = sqrt((xcen4-xw(k2g))**2.+(ycen4-yw(k2g))**2.+
     &                                   (zcen4-zw(k2g))**2.) 
                    rad2 = sqrt((xcen4-xw(k3g))**2.+(ycen4-yw(k3g))**2.+
     &                                   (zcen4-zw(k3g))**2.) 
                    rad3 = sqrt((xcen4-xw(k4g))**2.+(ycen4-yw(k4g))**2.+
     &                                   (zcen4-zw(k4g))**2.)
                            r4 = sqrt(xcen4**2. + ycen4**2. + zcen4**2.)
                                    dec4 = asin(zcen4/r4)*180./pi
                                    ra4 = atan(ycen4/xcen4)*180./pi

c                    write(*,*) xcen3, ycen3, zcen3, sep, rad, rad2, rad3
                                    
c     check conversion to ra and dec with various parts
                          if(ycen4.GT.0.AND.xcen4.LT.0) ra4 = ra4 + 180.
                          if(ycen4.LT.0.AND.xcen4.LT.0) ra4 = ra4 + 180.
c     check this is still in the survey
                          inout = mask(int(ra4)+1,int(dec4)-decoffset)
c                                    call checkdr7(ra4,dec4,inout)
                                    if(inout.EQ.1) then
                                       
                                      if(r4.LT.dmax.AND.r4.GT.dmin) then ! ok in survey

                                          diff = abs(rad2 - rad3)
              if(diff.GT.0.1) write(*,*) 'problem', sep, rad, rad2, rad3
                         if(diff.GT.0.1) write(*,*) 'problem', ra4, dec4             
                                          nh = nh + 1
                                          xvd(nh) = xcen4
                                          yvd(nh) = ycen4
                                          zvd(nh) = zcen4
                                          rvd(nh) = rad
                                          ravd(nh) = ra4
                                          decvd(nh) = dec4
                                          dvd(nh) = r4
                                       end if
                                    end if
                                 end if
                              end if
                           end if
                        end if
                     end if
                  end if
               end if
            end do
         end do
      end do
      
      write(*,*) 'found a total of ', nh, 'potential voids'

c need to sort the holes into size order

      call sort_4(nh,rvd,xvd,yvd,zvd,wksp,iwksp)

      do i = 1, nh
c rearrange sort to have largest first
        npvh = i
        xh(npvh) = xvd(nh-i+1)
        yh(npvh) = yvd(nh-i+1)
        zh(npvh) = zvd(nh-i+1)
        rh(npvh) = rvd(nh-i+1)
      end do
      npvh = nh

      npv = 0
      do i = 1, npvh
         if(mod(i,1000).EQ.0) write(*,*) i
         ncheck = 0
         ncheck2 = 0
         do k = 1, 10000
            x = ran3(seed)*rh(i)*2 + xh(i) - rh(i)
            y = ran3(seed)*rh(i)*2 + yh(i) - rh(i)
            z = ran3(seed)*rh(i)*2 + zh(i) - rh(i)
            diff = sqrt((x-xh(i))**2.+(y-yh(i))**2.+(z-zh(i))**2.)
            if(diff.LT.rh(i)) then
               ncheck = ncheck + 1
               r = sqrt(x**2. + y**2. + z**2.)
               decf = asin(z/r)
               raf = atan(y/x)
c**** check the ra and pi thing here
               if(y.GT.0.AND.x.LT.0) raf = raf + pi
               if(y.LT.0.AND.x.LT.0) raf = raf + pi
               raf = raf*180./pi
               decf = decf*180./pi
c     check this is still in the survey
               inout = mask(int(raf)+1,int(decf)-decoffset)
c               call checkdr7(raf,decf,inout)
               if(inout.EQ.1) then

c               if(x.GT.xmin.AND.x.LT.xmax.AND.y.GT.ymin.AND.y.LT.ymax
c     &                 .AND.z.GT.zmin.AND.z.LT.zmax) then

***********CHANGED HERE TO INCLUDE DECMIN*****************
c     need to test if actually in SDSS
                  ncheck2 = ncheck2 + 1
               end if
            end if
         end do
c         write(*,*) ncheck2, ncheck, rh(i)
c         if(ncheck2.EQ.ncheck) then
         if(ncheck2.GT.0.9*ncheck) then ! this means that void is 90% in the survey
            npv = npv + 1
            xpv(npv) = xh(i)
            ypv(npv) = yh(i)
            zpv(npv) = zh(i)
            rpv(npv) = rh(i)
            dpv(npv) = sqrt(xh(i)**2. + yh(i)**2. + zh(i)**2.)
            rapv(npv) = atan(yh(i)/xh(i))
            decpv(npv) = asin(zh(i)/dpv(npv))
c ****** this is another case of ra and pi
            if(yh(i).GT.0.AND.xh(i).LT.0) rapv(npv) = rapv(npv) + pi
            if(yh(i).LT.0.AND.xh(i).LT.0) rapv(npv) = rapv(npv) + pi
            decpv(npv) = decpv(npv)*180/pi
            rapv(npv) = rapv(npv)*180/pi
            write(33,54) xh(i), yh(i), zh(i), rh(i), rapv(npv), 
     &                  decpv(npv), dpv(npv)
         end if
c         write(33,54) xh(i), yh(i), zh(i), rh(i), rapv(npv), 
c     &        decpv(npv), dpv(npv)
      end do
 54   format(F10.5,1x,F10.5,1x,F10.5,1x,F10.5,1x,F10.5,1x,
     &       F10.5,1x,F10.5)

      write(*,*) 'number of holes in survey is', npv

c the largest hole is a void      
      nv = 1
      xv(1) = xpv(1)
      yv(1) = ypv(1)
      zv(1) = zpv(1)
      rv(1) = rpv(1)

      do i = 2, npv
         flag(i) = 0
         if(rpv(i).GT.10) then   ! this is where we impose max 10Mpc
            ntouch = 0
            do j = 1, nv
               sep = sqrt((xpv(i)-xv(j))**2. + (ypv(i)-yv(j))**2. 
     &              + (zpv(i)-zv(j))**2.)
               diff = rpv(i) + rv(j)
               if(sep.LT.0.1) ntouch = 1
c     calculate the overlapping volume. 
               r1 = rpv(i)
               r2 = rv(j)
c     check r2 > r1
               if((r2-r1).LE.sep.AND.sep.LE.(r2+r1)) then
                  a1 = (sep**2. + r1**2. - r2**2.)/2./sep
                  a2 = sep - a1
                  v = 2*pi*(r1**3. + r2**3.)/3. - pi*(r1**2.*a1+r2**2*a2  
     &                 - a1**3./3 - a2**3./3)
                  vs = 4*pi*r1**3./3.
                  if(v.GT.frac*vs) then
c these two spheres overlap by more than X%. This sphere is not unique
                     ntouch = ntouch + 1
                     flag(i) = j
                  end if
               end if
            end do
c            write(*,*) i, ntouch
            if(ntouch.EQ.0) then
c this sphere doesn't overlap with any other previously found void significantly
               nv = nv + 1
               xv(nv) = xpv(i)
               yv(nv) = ypv(i)
               zv(nv) = zpv(i)
               rv(nv) = rpv(i)
            end if
         end if
      end do
      write(*,*) 'ngroup is', nv

      do i = 1, npv
         flag(i) = 0
      end do

      nh = 0
      do i = 1, npv
         ntouch = 0
         nspecial = 0
         do j = 1, nv
            sep = sqrt((xpv(i)-xv(j))**2. + (ypv(i)-yv(j))**2. 
     &           + (zpv(i)-zv(j))**2.)
            diff = rpv(i) + rv(j)
c calculate the overlapping volume. If greater than half the volume of the 
c smaller sphere then accept it. Otherwise don't.
            r1 = rpv(i)
            r2 = rv(j)
c check r2 > r1
            if(r2.NE.r1.AND.(r2-r1).LE.sep.AND.sep.LE.(r2+r1)) then
               a1 = (sep**2. + r1**2. - r2**2.)/2./sep
               a2 = sep - a1
               v = 2*pi*(r1**3. + r2**3.)/3. - pi*(r1**2.*a1 + r2**2*a2  
     &                          - a1**3./3 - a2**3./3)
               vs = 4*pi*r1**3./3.
               if(v.GT.0.5*vs) then
                  ntouch = ntouch + 1
                  flag(i) = j
               end if
            end if
            if(sep.EQ.0.) then
c this is the hole that is the void and needs to be included too
               nspecial = 1
               flagg(i) = j
            end if
         end do
         if(nspecial.EQ.1) then 
            nh = nh + 1
            xh(nh) = xpv(i)
            yh(nh) = ypv(i)
            zh(nh) = zpv(i)
            rh(nh) = rpv(i)
            flog(nh) = flagg(i)
         end if
         if(ntouch.EQ.1) then
            nh = nh + 1
            xh(nh) = xpv(i)
            yh(nh) = ypv(i)
            zh(nh) = zpv(i)
            rh(nh) = rpv(i)
            flog(nh) = flag(i)
         end if
      end do

      do i = 1, nv
         n(i) = 0
      end do

      do i = 1, nh
c         r4 = sqrt(xh(i)**2. + yh(i)**2. + zh(i)**2.)
c         dec4 = asin(zh(i)/r4)
c         ra4 = atan(yh(i)/xh(i))
c         if(yh(i).GT.0.AND.xh(i).LT.0) ra4 = ra4 + pi
c         if(yh(i).LT.0.AND.xh(i).LT.0) ra4 = ra4 + pi
         write(72,*) xh(i), yh(i), zh(i), rh(i), flog(i)
         n(flog(i)) = n(flog(i)) + 1
      end do

      ngroup = nv

c see if can think of a better way for this. Maybe iterative...
      nran = 10000
      do i = 1, ngroup
         nsph = 0
         box = rv(i)*4. 
         do j = 1, nran
            x = ran3(seed)*box + xv(i) - box/2.
            y = ran3(seed)*box + yv(i) - box/2.
            z = ran3(seed)*box + zv(i) - box/2.
            test = 0
            do k = 1, nh
               if(flog(k).EQ.i.AND.test.LT.1) then
c     calculate difference between particle and sphere
                  sep = sqrt((x-xh(k))**2.+(y-yh(k))**2.+(z-zh(k))**2.)
                  if (sep.LT.rh(k)) then
c     this particles lies in at least one sphere 
                     test = 1
                     nsph = nsph + 1
                     r = sqrt(x**2. + y**2. + z**2.)
                     decv = asin(z/r)
                     rav = atan(y/x)
                     if(y.GT.0.AND.x.LT.0) rav = rav + pi
                     if(y.LT.0.AND.x.LT.0) rav = rav - pi
                     rav = rav*180./pi
                     decv = decv*180./pi
c these particles make up the void area
c                     write(77,*) r, rav, decv
                  end if
                end if
             end do
          end do
        voidvol(i) = box**3.*float(nsph)/float(nran)
c        write(*,*) voidvol(i), nsph, box, nran
       end do

       
       do i = 1, nh
          nfield(i) = 0
       end do

       write(*,*) nh
       do j = 1, nf
          test = 0
          do i = 1, nh
             if(test.EQ.0) then
                sep = sqrt((xf(j)-xh(i))**2.+(yf(j)-yh(i))**2.+
     &               (zf(j)-zh(i))**2.)
c     write(*,*) sep, rh(i)

                if (sep.LT.rh(i)*0.99) then
                   nfield(flog(i)) = nfield(flog(i)) + 1
c     make sure only count each field galaxy once
                   test = 1
                   write(83,*) xf(j), yf(j), zf(j), flog(i)
                end if
             end if
          end do
       end do

       nin = nwall + nf

       vol = 18610000. !****** volume of SDSS

       
       do i =  1, ngroup
c          write(*,*) nfield(i), voidvol(i), nin, vol
          deltap = (nfield(i) - nin*voidvol(i)/vol)/(nin*voidvol(i)/vol)
          sep = (voidvol(i)*3/4/pi)**0.3333333
          r = sqrt(xv(i)**2. + yv(i)**2. + zv(i)**2.)
          decv = asin(zv(i)/r)*180/pi
          rav = atan(yv(i)/xv(i))*180/pi
          if(yv(i).GT.0.AND.xv(i).LT.0) rav = rav + 180
          if(yv(i).LT.0.AND.xv(i).LT.0) rav = rav + 180
          rad = 4*pi*rv(i)**3./3./voidvol(i)
          write(12,23) xv(i),yv(i),zv(i),rv(i),r,rav,decv
          write(9,202) rv(i), sep, voidvol(i),r,rav,decv,deltap,
     &        nfield(i),rad
       end do
 202   format(f6.3,1x,f6.3,1x,f10.3,1x,f8.3,1x,f8.3,1x,f8.3,1x,
     &      f6.3,i5,1x,1x,f6.3)
       
 23    format(F10.5,1x,F10.5,1x,F10.5,1x,F10.5,1x,F10.5,1x,
     &       F10.5,1x,F10.5)



c finally go back to the maglim catalog and see which galaxies lie in a void

      open(37,file='Mzobs.tab')

      ngb = 0

      omega0 = 0.27
      lambda0 = 0.73
      beta = 0
      dlum = 0
      dlzcomb = 0
      call tabulate_dist(omega0,lambda0,beta)

      do i=1,10000000
         if(mod(i,1000).EQ.0) write(*,*) i
         read (37,*,end=87) rai,deci,lambda,eta,zi,rabsmagi,uri
c         write(*,*) i, rai,deci,lambda,eta,zi,rabsmagi,uri
         inout = mask(int(rai)+1,int(deci)-decoffset)
c         call checkdr7(rai,deci,inout)
c check in the boundary we are going to use using inout
         if(inout.EQ.1) then


            zplus1 = zi + 1
            call search_dist(r,zplus1,dlum,dlzcomb,2)
            if(r.GT.dmin.AND.r.LT.dmax) then
               ngb = ngb + 1
               xf(ngb) = r*cos(rai*pi/180.)*cos(deci*pi/180.)
               yf(ngb) = r*sin(rai*pi/180.)*cos(deci*pi/180.)
               zf(ngb) = r*sin(deci*pi/180.)
c               write(52,*) xf(ngb), yf(ngb), zf(ngb) 
            end if
         end if
      end do
 87   write(*,*) ngb


       do i = 1, nh
          nfield(i) = 0
       end do

       do j = 1, ngb
          test = 0
          do i = 1, nh
             if(test.EQ.0) then
                sep = sqrt((xf(j)-xh(i))**2.+(yf(j)-yh(i))**2.+
     &               (zf(j)-zh(i))**2.)
c     write(*,*) sep, rh(i)
                if (sep.LT.rh(i)*0.99) then
c     make sure only count each mag lim galaxy once
                   test = 1
c                   write(84,*) xf(j), yf(j), zf(j), flog(i)
                end if
             end if
          end do
       end do

c$$$c Another test
c$$$
c$$$      open(38,file='good_maglim.dat')
c$$$      open(85,file='good_maglim_voidgals.dat')
c$$$
c$$$      ngb = 0
c$$$
c$$$      do i=1,10000000
c$$$         read(38,*,end=88) x, y, z
c$$$         r = sqrt(x**2. + y**2. + z**2.)
c$$$         if(r.GT.dmin.AND.r.LT.dmax) then
c$$$            ngb = ngb + 1
c$$$            xf(ngb) = x
c$$$            yf(ngb) = y
c$$$            zf(ngb) = z
c$$$         end if
c$$$      end do
c$$$ 88   write(*,*) ngb
c$$$
c$$$       do i = 1, nh
c$$$          nfield(i) = 0
c$$$       end do
c$$$
c$$$       do j = 1, ngb
c$$$          test = 0
c$$$          do i = 1, nh
c$$$             if(test.EQ.0) then
c$$$                sep = sqrt((xf(j)-xh(i))**2.+(yf(j)-yh(i))**2.+
c$$$     &               (zf(j)-zh(i))**2.)
c$$$c     write(*,*) sep, rh(i)
c$$$                if (sep.LT.rh(i)*0.99) then
c$$$c     make sure only count each mag lim galaxy once
c$$$                   test = 1
c$$$                   write(85,*) xf(j), yf(j), zf(j), flog(i)

c$$$                end if
c$$$             end if
c$$$          end do
c$$$       end do
c$$$

  
      end
      

c-------------------------------------------------------------------
c----- this part is for dr7 sample
      subroutine checkdr7(ra,dec,inout)

      inout = 0
      if (dec.ge.0.and.dec.lt.5)then
         if (ra.ge.122.93.and.ra.le.249.95)then
            inout = 1
         end if
      else if (dec.ge.5.and.dec.lt.10)then
         if (ra.ge.119.91.and.ra.le.248.15)then
            inout = 1
         end if
      else if (dec.ge.10.and.dec.lt.15)then
         if (ra.ge.117.43.and.ra.le.251.36)then
            inout = 1
         end if
      else if (dec.ge.15.and.dec.lt.20)then
         if (ra.ge.115.56.and.ra.le.256.59)then
            inout = 1
         end if
      else if (dec.ge.20.and.dec.lt.25)then
         if (ra.ge.114.20.and.ra.le.258.34)then
            inout = 1
         end if
      else if (dec.ge.25.and.dec.lt.30)then
         if (ra.ge.112.91.and.ra.le.261.57)then
            inout = 1
         end if
      else if (dec.ge.30.and.dec.lt.35)then
         if (ra.ge.113.07.and.ra.le.261.65)then
            inout = 1
         end if
      else if (dec.ge.35.and.dec.lt.40)then
         if (ra.ge.110.56.and.ra.le.258.78)then
            inout = 1
         end if
      else if (dec.ge.40.and.dec.lt.45)then
         if (ra.ge.110.and.ra.le.255.47)then
            inout = 1
         end if
      else if (dec.ge.45.and.dec.lt.50)then
         if (ra.ge.112.1.and.ra.le.251.95)then
            inout = 1
         end if
      else if (dec.ge.50.and.dec.lt.55)then
         if (ra.ge.116.16.and.ra.le.246.79)then
            inout = 1
         end if
      else if (dec.ge.55.and.dec.lt.60)then
         if (ra.ge.119.97.and.ra.le.240.52)then
            inout = 1
         end if
      else if (dec.ge.60.and.dec.lt.65)then
         if (ra.ge.126.96.and.ra.le.232.48)then
            inout = 1
         end if
      else if (dec.ge.65.and.dec.lt.70)then
         if (ra.ge.136.04.and.ra.le.216.70)then
            inout = 1
         end if
      end if
      
      return
      end



