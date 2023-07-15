c
      subroutine lfneweff_sk6(vertex, dir, effhit, watert, log_level, is_mc)
      implicit none
c
c   (purpose)
c      calcurate effective number of hits for solar neutrino analysis
c
c   (input)
c      vertex : vertex
c      dir    : direction
c      watert : gain corrected water transparency
c      log_level : loglevel of findconst
c      is_mc : set is_mc = 1 for MC. Otherwise it will assume real data
c     
c   (output)
c      effhit : effhit(1)     gain corrected total effective hit at watert
c             : effhit(2:7)   gain corrected effective hit for PMT group 0-5 at watert
c             : effhit(8:12)  not used (=0.0)
c
c   occupancy correction  log(1./(1.-x))/x
c
c   (creation and data)
c   21-JUL-2006 ported from lfneweff_sk2 by Y.Takeuchi
c       o ***** tentative version *****
c       o use qetable3_0.dat in getnambu() for SK3
c       o others are not checked yet 
c   12-OCT-2006 ported from lfneweff_sk2 by Y.Takeuchi
c       o ***** tentative version *****
c       o modified totpmt from 5182 to 11129
c   16-OCT-2006 ported from lfneweff_sk2 by B.Yang
c       o ***** tentative version *****
c       o changed skcoverage_2 -> skcoverage_3
C   30-OCT-2009 replace lfoccor_2 with lfoccor_3 by S.Yamada
c
C   27-MAR-2017 implement Neff correction with gain increasing by Linyan
C   08-MAY-2017 replaced qetable4_0 with qetable4_1 by Linyan/Y.Takeuchi
c
c   11-NOV-2017 implement different Neff correction factor for MC
c   21-APR-2020 implement Shinoki's Neff calculation by y.takeuchi
c     o lfneweff_sk4.F -> lfneweff_sk4_final.F
c     o lfhitgeom.F -> lfhitgeom_sk4_final.F
c     o skcoverage_3.F -> skcoverage_sk4_final.F
c
c   4-JUN-2020 replaced qetable4_1 with qetable4_3 by M.Shinoki
c
c   24-SEP-2021 update lfneweff_sk4.F -> lfneweff_sk5.F by M.Shinoki
c     o "WITHOUT" gain correction 
c     o Water parameters for SK-V
c     o beta <-> TBA correlation function is updated for SK-V
c     o qetable4_3 -> qetable5_2
c
c   01-Oct-2021 update lfneweff_sk5.F by M.Shinoki
c     o add the subroutine "getprodytable" to read PMT producted year
c   24-Oct-2022 update lfneweff_sk6.F by K.Ieki
c     o Updated based on lfneweff_sk4_final_qe43 with new parameter 
c       values for SK6.
c     o Added flag_g4_wtpar to select water parameter from 
c       detsim or G4.
c      
      
     
#include "skparm.h"
#include "skhead.h"
#include "sktq.h"
#include "geopmt.h"
c#include "skbadc0.h"
#include "skbadc.h"
c 2020/04/03 M.Shinoki added "skday.h"
#include "skday.h"
#include "lfparm.h"
      
      integer totpmt, irun, loglv, log_level, is_mc
cc      parameter (totpmt = 5182)

ccyt 2017/04/13 skmisch add missing channels as badch. so, totpmt should be 11146(=MAXPM)?
c      parameter (totpmt = 11129)
      parameter (totpmt = MAXPM)

      real conver
      real conver_data
      parameter (conver_data = 0.0)
c      parameter (conver_data = 0.226)
      real conver_mc_sk2pmt
      real conver_mc_sk3pmt
c      parameter (conver_mc_sk2pmt = 0.131)
c      parameter (conver_mc_sk3pmt = 0.149)
      parameter (conver_mc_sk2pmt = 0.0)
      parameter (conver_mc_sk3pmt = 0.0)

      integer ihitcab(maxpm)
      real vertex(3),dir(3),effhit(12)
      integer ihitflag(maxpm)
      real dist, costh, cospm, theta, phi, dark, darkc,tailcor
      real xoccor, dum, darkcor
      real darkc_all, darkc_nhit_ave
      integer nhitt, i0, j0
      real t(maxpm), watert
ccyt not used in this version (different definition)
c     real xlambda(12)
c      data xlambda/ 7000., 8000., 9000., 10000., 11000.
c     &     , 12000., 13000., 14000., 15000., 16000., 17000., 18000./
      real ttail, twin
      real skcoverage_sk4_final, skcoverage_barrel
      real coseffsk_2
      real badcor
      integer nhit
      parameter (ttail=100.,twin=50.)
      integer i,j,k
      real tmpval
c   Nambu table correction
      real xnambu(maxpm)
      save xnambu
      integer icall
      data icall/0/
      save icall

      real gain5(5)
      save gain5
      integer pmtgain(maxpm)
      save pmtgain
      integer prodytable(maxpm)
      save prodytable
      integer currentrun
      save currentrun
      data currentrun/-1/
      real sk4gain(5)
      data sk4gain/0.992057, 1.00184, 0.989083, 0.98216, 0.987678/
      save sk4gain
      real igaincab(maxpm)

      integer loc,ierr,ndata,idata(18)
      real rdata(18)
      equivalence (idata(1),rdata(1))

c     Definition parameters M.Shinoki 2020/03/25
      real wt_eff
      integer fileline
      parameter (fileline = 400)
      real lambda_tmp, weight_tmp, popefrysg_tmp
      real wavelength(fileline), weight(fileline), popefrysg(fileline)
      real alpha_abs(fileline), alpha_sca(fileline)
      real alpha_symsc, alpha_asysc
      real beta, TBA
      real z_border
      real absfitfac, absfit, absfac, abssfac, A1, abspow
      real rayfac, raysfac, asyfit, miefac, miesfac
      parameter (z_border = -1100.)
      parameter (A1 = 0.0279)      

      integer icase
      real z_vtx_pmt, z_vtx_border, dist_vtx_border
      real blamda_ms, w_ms
      real surv_prob, surv_prob_1, surv_prob_2
      real cross_section, cross_section_1, cross_section_2

c     Scaling factor of scattering
      real const_scat

c     Wavelength to switch to Pope Fry
      real lambda_popefry

      character*200 file_name


      if (icall.eq.0) then
         print *, 'lfneweff_sk6: log_level = ', log_level
         icall = icall + 1
      endif
     
      if (nrunsk.ne.currentrun) then

c     Set water parameters
         if (flag_g4_wtpar.eq.1) then
            ! G4 water parameters
            absfit  = 0.5966      ! P0
            absfac  = 5.18888E+7  ! P1 
            abssfac = 1.06522     ! P2
            abspow  = 14.1858     ! P3
            rayfac  = 1.13817E+08 ! P4
            raysfac = 57910.8     ! P5
            asyfit  = 2.26159E-04 ! P6
            miefac  = 17.126      ! P7
            miesfac = 44862.2     ! P8
            ! const_scat = 0.4423
            const_scat = 0.4998
            lambda_popefry = 533.9
         else 
            ! detsim water parameters
            absfit  = 0.6580      ! P0
            absfac  = 4.43153E+7  ! P1
            abssfac = 1.181139    ! P2
            abspow  = 15.3261     ! P3
            rayfac  = 1.12401E+08 ! P4
            raysfac = 67368.4     ! P5
            asyfit  = 1.87782E-04 ! P6
            miefac  = 3.18720E+5  ! P7
            miesfac = 613.511     ! P8
            const_scat = 0.6194   
            lambda_popefry = 485.1
         endif
         if (flag_g4_wtpar.eq.1) then
            print *, 'Use G4 water parameters'
         else
            print *, 'Use detsim water parameters'
         endif
         print *, 'const_scat = ', const_scat
         print *, 'Water pars: SK-VI, absfac=', absfac, 'rayfac=', rayfac, 'miefac=', miefac

         if (log_level.ge.1 .and. log_level.le.5) then
            if (icall.eq.10.and. log_level.eq.4) then
               print *, 'lfneweff_sk6: suppress log'
               loglv = 4
               icall = icall + 1
            else if (icall.lt.10 .and. log_level.eq.4) then
               loglv = 5
               icall = icall + 1
            else 
               loglv = log_level
            endif
         else
            loglv = 5
         endif
         
         call getgain(sk4gain, gain5, igaincab, prodytable,loglv)
c         call getprodytable(prodytable,loglv)

         irun=nrunsk
15       call skdark(irun, i)
         if (i.ne.0) then  ! within 20runs
           irun = irun -1
           goto 15
         endif

c******* merge lftba and calclate beta into lfgettba
c******* 2022/10/28 M.Harada, Nakano
#ifdef TENTATIVE
         beta = 0
#else
         call lfgettba(nrunsk, watert, 3, beta, 1) ! 3: with gain_corr
c        **************************************************************
#endif
c***** absfitfac and beta evaluation ***** 2020/04/03 M.Shinoki
c*** absfitfac/absfit ***
         call trabs_sk6(watert,absfitfac)
         print *, 'absfitfac = ', absfitfac, 'watert = ', watert
c        **************************************************************

         call getsilly(xnambu)

c        ***** Input wavelength/weight/popefrysg ***** M.Shinoki 2020/03/25
         call findconsts(file_name,'lambda_weight_popefrysg.dat',
     $        'lambda_weight_popefrysg.dat',2)
         open(95, file=file_name, form='FORMATTED', err=999)
         do i=1,fileline
            read(95,*) lambda_tmp, weight_tmp, popefrysg_tmp
            wavelength(i) = lambda_tmp
            weight(i) = weight_tmp
            popefrysg(i) = popefrysg_tmp
         end do
         close(95)
c        ***** End input wavelength/weight/popefrysg *****
         
c     ***** Calculation alpha abs/sca ***** M.Shinoki 2020/04/07
         do i=1,fileline
            blamda_ms = wavelength(i)
            ! alpha absorption
            if(blamda_ms < lambda_popefry)then
               alpha_abs(i) = absfitfac*absfit*absfac/blamda_ms**4 + absfitfac*absfit*abssfac*A1*(blamda_ms/500)**abspow
            endif
            if(blamda_ms > lambda_popefry)then
               alpha_abs(i) = absfitfac*absfit*absfac/blamda_ms**4 + popefrysg(i)
            endif
            ! alpha scattering
            alpha_symsc = (rayfac/blamda_ms**4) * (1.0 + raysfac/blamda_ms**2)
            alpha_asysc = asyfit*(1.0 + (miefac/blamda_ms**4)*(blamda_ms-miesfac)**2)
            alpha_sca(i) = alpha_symsc + alpha_asysc
            ! [1/m] -> [1/cm]
            alpha_abs(i) = alpha_abs(i)/100.
            alpha_sca(i) = alpha_sca(i)/100.
                     
            ! alpha_abs correction
            alpha_abs(i) = alpha_abs(i)/(1+69.63*beta)

            ! alpha_sca correction    2020/04/19 M.Shinoki
            alpha_sca(i) = alpha_sca(i)*const_scat
         enddo
c     ***** End Calculation alpha abs/sca *****

         currentrun = nrunsk

c     ***** End parameter settings for each run *****
      endif

c-----------------------------------------------------------------------
c     read and store nambu table & Shoniki's table
c-----------------------------------------------------------------------

      do i=1,12
          effhit(i)=0.
      end do

c  if vertex is outside of inner volume, exit
      if(sqrt(vertex(1)**2+vertex(2)**2).gt.1690.
     &   .or. abs(vertex(3)).gt.1810.)goto 1000

c  bad channel correction factor
      badcor=float(totpmt)/float(totpmt-nbad0)

c  calculate version 2 ********************************************
c   pick up hit cables
      call lfnhit2(ttail,vertex,nhitt,ihitcab,t)
      call lfmaxrgn(t,nhitt,twin,i0,j0)
      nhit=j0-i0+1
      do i=i0,j0
        ihitcab(i-i0+1)=ihitcab(i)
        t(i-i0+1)=t(i)
      end do
c
c   mark selected cables
      do i=1,maxpm
        ihitflag(i)=0
      end do
      do i=1,nhit
        ihitflag(ihitcab(i))=1
      end do

C      stop

      darkc_nhit_ave = 0.
      do i=1,nhit
         darkc_nhit_ave = darkc_nhit_ave + dark_rate(ihitcab(i))
      end do
      darkc_nhit_ave = darkc_nhit_ave / nhit
      
      do i=1,nhit
	if (xnambu(ihitcab(i)) .le. 0.) then
c	if (xnambu(ihitcab(i) .le. 0.)) then
	  write(*,*) 'lfneweff_sk6: hit exists in missing PMT!!'
	  write(*,*) '            PMT cableID=', ihitcab(i)
	  goto 10
	end if
        call lfhitgeom_sk4_final(vertex,dir,ihitcab(i),dist,costh,cospm,theta,phi)
        if(dist.eq.0.) goto 10

        conver = conver_data
c     find out PMT type in the case of MC
        if (is_mc .eq. 1) then
           if (prodytable(ihitcab(i)) .eq. 1
     &          .or. prodytable(ihitcab(i)) .eq. 2) then ! SK2 PMT
              conver = conver_mc_sk2pmt
           else if (prodytable(ihitcab(i)) .eq. 3
     &             .or. prodytable(ihitcab(i)) .eq. 4
     &             .or. prodytable(ihitcab(i)) .eq. 5) then ! SK3 PMT
              conver = conver_mc_sk3pmt
           else
              conver = 0.0
           endif
        endif

        
c   obtain dark rate and dark current correction factor
c      call lfsetdark(nrunsk,dark)
c     darkc_all = dark*float(totpmt-nbad0)*1.e-9
      darkc_all = dark_ave *float(totpmt-nbad0)*1.e-6
c      write(*,*)'lfsetdark: ', darkc


c   obtain correction from hits outside of narrower window
      tailcor=float(nhitt-nhit)-darkc_all*(ttail-twin)
      dum=tailcor/float(nhit)
      if(tailcor.lt.0.)tailcor=0.
      if(tailcor.gt.float(nhit)*0.1)tailcor=float(nhit)*0.1

      
c     correction factor for dark current and tail hits
      darkcor=(darkc_all*twin * dark_rate(ihitcab(i))/darkc_nhit_ave-tailcor)/float(nhit)

c      write(*,*)'lfsetdark: ',ihitcab(i), dark, dark_rate(ihitcab(i)), darkc_all, darkc_nhit_ave, darkcor, tailcor

c  obtain occupancy correction
C        call lfoccor_2(ihitcab,ihitflag,ihitcab(i),xoccor)
c        call lfoccor_3(ihitcab,ihitflag,ihitcab(i),xoccor)
        call lfoccor_4(ihitcab,ihitflag,xnambu,ihitcab(i),xoccor)  
 
c  occupancy, dark current, effective pmt density, transparency corrections
c	tmpval = (xoccor - darkcor) /
c        write(*,*) 'Now checking the correction: ', igaincab(ihitcab(i)), conver
c        write(*,*) xoccor, darkcor

c     ***** New skcoverage for barrel and cap ***** M.Shinoki
        if(ihitcab(i).le.7650) then
         tmpval = (xoccor - darkcor) /(1+igaincab(ihitcab(i))*conver) /
     &          skcoverage_barrel(theta, phi) /
     &          xnambu(ihitcab(i))
        else if(ihitcab(i).ge.7651) then
         tmpval = (xoccor - darkcor) /(1+igaincab(ihitcab(i))*conver) /
     &          skcoverage_sk4_final(theta, phi) /
     &          xnambu(ihitcab(i))
        endif
c     ***** End New skcoverage for barrel and cap *****

c     ***** Effective water transparency calculation ***** M.Shinoki 2020.04.07
        wt_eff = 0.
        
        z_vtx_pmt = xyzpm(3,ihitcab(i))-vertex(3)
        z_vtx_border = z_border-vertex(3)
        dist_vtx_border = dist*abs(z_vtx_border)/abs(z_vtx_pmt)

c     Classification of photon track
        if(vertex(3)>z_border.and.xyzpm(3,ihitcab(i))>z_border)then
           icase = 1
        endif
        if(vertex(3)>z_border.and.xyzpm(3,ihitcab(i))<z_border)then
           icase = 2
        endif
        if(vertex(3)<z_border.and.xyzpm(3,ihitcab(i))<z_border)then
           icase = 3
        endif
        if(vertex(3)<z_border.and.xyzpm(3,ihitcab(i))>z_border)then
           icase = 4
        endif

c     Integrate about photon-wavelength
c     Evaluate survival probability of photon
        surv_prob = 0.
        do k=1,fileline
           w_ms = weight(k)

           if(icase.eq.1)then
              cross_section = alpha_abs(k)*(1+beta*(vertex(3)+0.5*z_vtx_pmt))+alpha_sca(k)
              surv_prob = surv_prob + w_ms*exp(-cross_section*dist)
           endif
           if(icase.eq.2)then
              cross_section_1 = alpha_abs(k)*(1+beta*(vertex(3)+0.5*z_vtx_border))+alpha_sca(k)
              cross_section_2 = alpha_abs(k)*(1+beta*z_border)+alpha_sca(k)
              surv_prob = surv_prob + w_ms*exp(-cross_section_1*dist_vtx_border)*exp(-cross_section_2*(dist-dist_vtx_border))
           endif           
           if(icase.eq.3)then
              cross_section = alpha_abs(k)*(1+beta*z_border)+alpha_sca(k)
              surv_prob = surv_prob + w_ms*exp(-cross_section*dist)
           endif
           if(icase.eq.4)then
              cross_section_1 = alpha_abs(k)*(1+beta*z_border)+alpha_sca(k)
              cross_section_2 = alpha_abs(k)*(1+beta*(z_border+0.5*(z_vtx_pmt-z_vtx_border)))+alpha_sca(k)
              surv_prob = surv_prob + w_ms*exp(-cross_section_1*dist_vtx_border)*exp(-cross_section_2*(dist-dist_vtx_border))
           endif
        enddo
        wt_eff = -dist/log(surv_prob)
c     ***** End of effective water transparency calculation *****
        
        effhit(1)=effhit(1) + tmpval * exp(dist/wt_eff)
        do j=2,7
          if(prodytable(ihitcab(i)).eq.j-2)then
             effhit(j)=effhit(j) + tmpval * exp(dist/wt_eff)
          endif
        enddo
ccyt don't fill 8-12
c        do j=8,12
c          effhit(j)=effhit(j)+tmpval*exp(dist/xlambda(j)) 
c        enddo
 10     continue
      end do
      do i=1,12
        effhit(i)=effhit(i)*badcor
      end do

#if defined(DEBUG)
      write(*,*) 'Neff ', effhit(1),effhit(2),effhit(3),
     &           effhit(4), effhit(5), effhit(6),
     &           effhit(7), effhit(8), effhit(9),
     &           effhit(10), effhit(11), effhit(12)
#endif
c
 1000 continue

c   store results in the bank
c      call kzbloc('LFEFF2',loc)
c      if(loc.eq.0)then
c         call kzbcr1('LFEFF2',1,'1I',1,18,'3I15F',ierr)
c      else
c         call kzget1('LFEFF2',1,ndata,rdata)
c      endif
c      if(effhit(1).gt.0)then
c         idata(1)=nhit
c         idata(2)=nhitt
c         idata(3)=nbad0
c         rdata(4)=dark
c         rdata(5)=dum
c         rdata(6)=darkcor
c         rdata(7)=effhit(1)
c         rdata(8)=effhit(2)
c         rdata(9)=effhit(3)
c         rdata(10)=effhit(4)
c         rdata(11)=effhit(5)
c         rdata(12)=effhit(6)
c         rdata(13)=effhit(7)
c         rdata(14)=effhit(8)
c         rdata(15)=effhit(9)
c         rdata(16)=effhit(10)
c         rdata(17)=effhit(11)
c         rdata(18)=effhit(12)
c      else
c         idata(1)=0
c         idata(2)=0
c         idata(3)=0
c         rdata(4)=0.
c         rdata(5)=0.
c         rdata(6)=0.
c         rdata(7)=0.
c         rdata(8)=0.
c         rdata(9)=0.
c         rdata(10)=0.
c         rdata(11)=0.
c         rdata(12)=0.
c         rdata(13)=0.
c         rdata(14)=0.
c         rdata(15)=0.
c         rdata(16)=0.
c         rdata(17)=0.
c         rdata(18)=0.
c      end if
c      call kzrep1('LFEFF2',1,18,rdata)

      return

 999  continue
      write(*,*) 'lfneweff_sk6: cannot open file:'
      write(*,*) file_name
      stop
      
      end

c***********************************************************************
c***********************************************************************
      subroutine getsilly(xnambu)
      implicit none

#include "skparm.h"

      integer ipmt, i
      integer lun
      parameter (lun=91)
      character*200 fname
      real xnambu(maxpm)
      real factora(maxpm), factorb(maxpm)

c      call findconsts(fname, 'Nambu gaintable',
c     &  'gaintable_water.dat',0)
ccyt     &  'qetable_1.dat',0)
ccyt     &  'qetable3_0.dat',0)

C      call findconsts(fname, 'qetable', 'qetable5_2.dat',2)
      call findconsts(fname, 'qetable', 'qetable5_3.dat',2)
      print *, 'QE data:', fname
      open(unit=lun, file=fname, status='old',form='formatted')

      do ipmt=1,maxpm
ccyt	read(lun, *, end=211) factora(ipmt), factorb(ipmt)
ccyt	xnambu(ipmt)=(factora(ipmt)+factorb(ipmt))/2.0
	read(lun, *, end=211) i, factorb(ipmt)
	xnambu(ipmt)=factorb(ipmt)
      end do
 211  continue
      close(lun)

      return
      end

      
c***********************************************************************
c***********************************************************************
      subroutine getgain(sk4gain, gain5, igaincab, prodytable, loglv)
      implicit none

#include "skparm.h"
#include "skhead.h"

      integer readrun, loglv
      character*200 rskip(15)
      real gain5(5)
      integer pmtid, i
      real prodyear
      character*200 pskip(4)
      real igaincab(maxpm)
      integer prodytable(maxpm)
      real sk4gain(5)
      character*200 fname

      call findconsts(fname, 'water.ave10.2', 'water.ave10.2',loglv)
      open(93,file=fname,status='old',form='formatted')

      readrun = 0
      do while(readrun.lt.nrunsk)
         read(93, *) readrun, rskip, gain5
      enddo
      close(93)

      call findconsts(fname, 'pmt_prod_year_sk5.dat', 'pmt_prod_year_sk5.dat',loglv)
      open(92,file=fname,status='old',form='formatted')
      do i=1,MAXPM
         read(92,*) pmtid, prodyear, pskip
         if(prodyear.lt.1000)then
            igaincab(pmtid) = 0.
            prodytable(pmtid) = 0
         else if(prodyear.lt.1996)then
            igaincab(pmtid) = (gain5(1)-sk4gain(1))/sk4gain(1)
            prodytable(pmtid) = 1
         else if(prodyear.lt.1998)then
            igaincab(pmtid) = (gain5(2)-sk4gain(2))/sk4gain(2)
            prodytable(pmtid) = 2
         else if(prodyear.lt.2004)then
            igaincab(pmtid) = (gain5(3)-sk4gain(3))/sk4gain(3)
            prodytable(pmtid) = 3
         else if(prodyear.lt.2005)then
            igaincab(pmtid) = (gain5(4)-sk4gain(4))/sk4gain(4)
            prodytable(pmtid) = 4
         else
            igaincab(pmtid) = (gain5(5)-sk4gain(5))/sk4gain(5)
            prodytable(pmtid) = 5
         endif
c         print*, 'pmt ', pmtid, prodyear, igaincab(pmtid)
      enddo
      close(92)
cc      print*, 'called gaintable'

      return
      end
