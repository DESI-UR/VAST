c------------------------------------------------------------------------------
c Tabulate redshift, z, luminosity distance, d_L, and the combination
c d_L^2 (1+z)^-beta against comoving distance, given Omega_0 and Lambda_0
c     
      subroutine  tabulate_dist(omega0,lambda0,beta)
*************************************variables*********************************
      implicit none
      integer ntable,NT,it,NNT,jt
      real EPS,beta
c      parameter(NT=1000,EPS=1.0e-04,NNT=10000)
      parameter(NT=10000,EPS=1.0e-04,NNT=10000)
      real rcomov(NT),z(NT),dl(NT),comb(NT),zplus1,x,om,atanhom,rchar
      real drc,zplus1t(NNT),rct(NNT),dz,intp,intt,h,omega0,lambda0,rcuml
      common /dist_table/ ntable,drc,rcomov,z,dl,comb
*******************************************************************************
      ntable=NT
      if (omega0.lt.0.0) then    !flags non-relativistic assumption
      else if (abs(omega0-1.0).le.EPS .and. (lambda0).le.EPS) then !omega=1
      else if (abs(omega0-1.0).gt.EPS .and. (lambda0).le.EPS) then !open
         om=sqrt(1.0-omega0)  !useful numbers in open model
         atanhom=0.5*log((1.0+om)/(1.0-om) )
      else if (abs(omega0+lambda0-1.0).lt.EPS ) then !flat with lambda
c        First tabulate rcomov as with redshift by integrating
         dz=10.0/real(NNT)
         rcuml=0.0
         rct(1)=0.0
         zplus1t(1)=1.0
         intp=0.0
         do it=2,NNT
            zplus1t(it)=1.0+real(it-1)*dz
            intt=1.0/sqrt(omega0*zplus1t(it)**3 + 1.0 - omega0)
            rcuml=rcuml+0.5*(intt+intp)*dz
            rct(it)=rcuml*3000.0
            intp=intt
         end do
      else
         stop 'Not programmed for this cosmology'
      end if

c
c     Loop over comoving distances and compute corresponding
c     redshift and luminosity distances...
      drc=6000.0/real(NT)
      do it=1,NT   !make table uniformly spaced in comoving distance.
         rcomov(it)=real(it-1)*drc
c
         if (omega0.lt.0.0) then
            zplus1=1.0 +rcomov(it)/3000.0
            z(it)= rcomov(it)/3000.0
            dl(it)=rcomov(it)
            comb(it)= dl(it)**2 * zplus1**(-beta)
         else if (abs(omega0-1.0).le.EPS .and. (lambda0).le.EPS) then !omega=1
            zplus1=1.0/(1.0-rcomov(it)/6000.0)**2
            z(it)=zplus1-1.0
            dl(it)=rcomov(it)*zplus1
            comb(it)=dl(it)**2 * zplus1**(-beta)
c
         else if (abs(omega0-1.0).gt.EPS .and. (lambda0).le.EPS) then !open
            x=tanh(atanhom-om*rcomov(it)/6000.0)
            zplus1=((om/x)**2-om**2)/omega0
            z(it)=zplus1-1.0
            rchar=(6000.0/(zplus1*omega0**2)) *
     &           (omega0*z(it)+(omega0-2.0)*(sqrt(1+omega0*z(it))-1.0))
            dl(it)=rchar*zplus1
            comb(it)=dl(it)**2 * zplus1**(-beta)
c
         else if (abs(omega0+lambda0-1.0).lt.EPS ) then !flat
c           look up redshift from temporary table
            jt=1
            do while (rcomov(it).ge.rct(jt)) 
               jt=jt+1
               if (jt.gt.NNT) stop
     &         'tabulate_dist(): rct() not tabulated to sufficient z'
            end do
            h=(rcomov(it)-rct(jt-1))/(rct(jt)-rct(jt-1))
            zplus1=zplus1t(jt-1)*(1.0-h) + zplus1t(jt)*h
            z(it)=zplus1-1.0
            dl(it)=rcomov(it)*zplus1
            comb(it)=dl(it)**2 * zplus1**(-beta)
c
         end if
      end do
      
      return
      end
c------------------------------------------------------------------------------
c Given either rcomov, z, dl or dl^2(1+z)^-beta look up the corresponding
c values of the other quantities. The integer isel specifies which argument
c is set on input.
c     
      subroutine  search_dist(rc,zplus1,dlum,dlzcomb,isel)
*************************************variables*********************************
      implicit none
      integer ntable,NT,it,isel,itlo,ithi
      parameter(NT=10000)
      real rcomov(NT),z(NT),dl(NT),comb(NT),rc,zplus1,dlum,dlzcomb
      real drc,h,rit,zt
      save itlo,ithi
      common /dist_table/ ntable,drc,rcomov,z,dl,comb
      data itlo/1/
      data ithi/1/
*******************************************************************************
      if (isel.eq.1) then  !rc set so lookup zplus1,dlum and dlum^2(1+z)^-beta
c      This is the fast/easy case as the table is equally spaced in rc 
         rit=1.0+rc/drc
         it = int(rit)
         h= rit-real(it) 
         if (it.ge.NT) stop 'search_dist() r beyond tabulated range'
         zplus1=1.0+ z(it)*(1.0-h)    +    z(it+1)*h
         dlum=       dl(it)*(1.0-h)  +    dl(it+1)*h
         dlzcomb=  comb(it)*(1.0-h)  +  comb(it+1)*h
c
      else if (isel.eq.2) then !zplus1 set 
c      Search for corresponding redshift
         zt=zplus1-1.0
         if (z(ithi).lt.zt .or. z(itlo).gt.zt ) then!short cut if zt close to
            itlo=1                                  !last call
            ithi=NT                                 !otherwise do binary search
            do while (ithi-itlo.gt.1) 
               it=(ithi+itlo)/2
               if(z(it).gt.zt)then
                  ithi=it
               else
                  itlo=it
               endif
            end do      
         end if
         h=(zt-z(itlo))/(z(ithi)-z(itlo))
         rc=     rcomov(itlo)*(1.0-h)  +rcomov(ithi)*h
         dlum=       dl(itlo)*(1.0-h)  +    dl(ithi)*h
         dlzcomb=  comb(itlo)*(1.0-h)  +  comb(ithi)*h
c
      else if (isel.eq.3) then !luminosity distance set 
c      Search for corresponding luminosity distance
         if (dl(ithi).lt.dlum .or. dl(itlo).gt.dlum ) then
            itlo=1                                  
            ithi=NT                                 
            do while (ithi-itlo.gt.1) 
               it=(ithi+itlo)/2
               if(dl(it).gt.dlum)then
                  ithi=it
               else
                  itlo=it
               endif
            end do      
         end if
         h=(dlum-dl(itlo))/(dl(ithi)-dl(itlo))
         rc=     rcomov(itlo)*(1.0-h)  +rcomov(ithi)*h
         zplus1=1.0+ z(itlo)*(1.0-h)    +    z(ithi)*h
         dlzcomb=  comb(itlo)*(1.0-h)  +  comb(ithi)*h
c
      else if (isel.eq.4) then !dl^2 zplus1^-beta set 
c      Search for corresponding  dlum^2 zplus1^-beta set 
         if (comb(ithi).lt.dlzcomb .or. comb(itlo).gt.dlzcomb ) then
            itlo=1                                  
            ithi=NT                                 
            do while (ithi-itlo.gt.1) 
               it=(ithi+itlo)/2
               if(comb(it).gt.dlzcomb)then
                  ithi=it
               else
                  itlo=it
               endif
            end do      
         end if
         h=(dlzcomb-comb(itlo))/(comb(ithi)-comb(itlo))
         rc=     rcomov(itlo)*(1.0-h)  +rcomov(ithi)*h
         zplus1=1.0+ z(itlo)*(1.0-h)    +    z(ithi)*h
         dlum=       dl(itlo)*(1.0-h)  +    dl(ithi)*h
c
      end if
      return
      end
c------------------------------------------------------------------------------








