c 
c     lowfit.F         02-NOV-2006     Y.Takeuchi
c
c     for SK-III (re)process1
c
c   input:  (arg1)  run number
c
c 
c
      program lowfit

      implicit none

#include "skhead.h"
#include "skparm.h"
#include "sktq.h"
#include "skroot_lowe.h"
#include "geopmt.h" 
#include "skday.h"
#include "skwt.h"
#include "skbadc.h"
#include "lininfo2.h"
#include "lfparm.h"

      integer lun
      parameter (lun=10)

      integer nrun, iret, ipbank, rlp,lfflag,iday
      real watert, rver
      character*200 fout,fname1,fname2
      character*200 tmpchar

      character tentative
      character*200 fTentativeConfig
      logical ntentative
      real watertTentative
c*** number of hit limit for clusfit
      integer i, narg
      integer NHIT_CUT
      parameter (NHIT_CUT = 400) 

      integer lecompte,n,nnrun
      real px, py, pz,width

      integer log_level,iskip

      character*200 temp
      integer it,nintoskip,nouttoskip
      parameter (nintoskip=3)
      parameter (nouttoskip=1)
      character*200 in_branches_to_skip(nintoskip)
      character*200 out_branches_to_skip(nouttoskip)

      data in_branches_to_skip /'ATMPD','UPMU','SECONDARY'/
      data out_branches_to_skip /'SECONDARY'/
!      data out_branches_to_skip /'TQLIST','ODTQLIST','SECONDARY'/

cccccccccccccccccccccccccccccccccccccccccccccccccc

      flag_g4_wtpar = 1 ! for SKG4
      ! flag_g4_wtpar = 0 ! for detsim

c*** set SK_GEOMETRY
      ! SK_GEOMETRY = 5
      SK_GEOMETRY = 6

      narg = iargc()
      print *, narg
      call getarg(1,tmpchar)
      read(tmpchar,*)nrun
      call getarg(2,tmpchar)
      fname2 = tmpchar
      call skroot_open(lun,fname2(1:lnblnk(fname2)))

      do i = 3, 3
         call getarg(i, tmpchar) 
         call skroot_set_input_file(lun,tmpchar(1:lnblnk(tmpchar)))
        print *, "input file  : ", tmpchar
      enddo
#ifdef TENTATIVE
c***  tentative paramter
      call getarg(4, tentative)
      print *, 'tentative is ', tentative
      read(tentative,*)ntentative
      if (ntentative) then
         call getarg(narg, fTentativeConfig)
         open(unit=1996, file=fTentativeConfig)
         read(1996,*)watertTentative
         print *, 'tentative watert = ', watertTentative
      endif
#endif
c*** init.
      call kzinit
      call geoset
      call skbadopt(55)
      call skoptn('31,30,25')

c*** Set bad channel
      call skrunday
c      call skwt
      call skwt_gain_corr

c*** check lininfo
      call linsum2(nrun)
      print *, 'lininfo2:', lin_mode(nrun),lin_trg(nrun),
     $     lin_x(nrun),lin_y(nrun),lin_z(nrun),
     $     lin_badrun(nrun), lin_runnum, lin_mom(nrun),
     $     lin_dark, darkfac
      px = lin_x(nrun)
      py = lin_y(nrun)
      pz = lin_z(nrun)

      call skbadch(lin_badrun(nrun), 1, iret)
               
      call lfwater(relapse(lin_badrun(nrun)), watert)
      print *, 'nrunsk, lin_badrun, watert = ',
     $     nrun, lin_badrun(nrun), watert
#ifdef TENTATIVE
      if (ntentative) then
         watert=watertTentative
      endif
#endif
      ! watert=11500.0

      
c*** init. bonsai 
      CALL cfbsinit(MAXPM,xyzpm)

! c*** skip any unneeded branches; this must be called before skroot_init
!       do it=1,nintoskip
!             temp = in_branches_to_skip(it)
!             call skroot_zero_branch(lun,0,temp(1:lnblnk(temp)))
!       enddo
!       do it=1,nouttoskip
!             temp = out_branches_to_skip(it)
!             call skroot_zero_branch(lun,1,temp(1:lnblnk(temp)))
!       enddo

c*** open skroot file
      
c      call skroot_open(lun,fname2(1:lnblnk(fname2)))
c      call skroot_set_input_file(lun,fname1(1:lnblnk(fname1)))
      call skroot_init(lun)

c*** main loop 
 10   CONTINUE

c*** read an event
      CALL SKREAD(lun,*1002,*1001,*10,*10)

      nrunsk = nrun
      if(mod(nevsk,1000).eq.0) write(6,*) nrunsk,nevsk,nqisk
      call lfclear_all()
      call lfallfit_sk6_mc(watert,NHITCUT,iskip,log_level,lfflag)
      ! call lfallfit_sk5_mc(watert,NHITCUT,iskip,log_level,lfflag)
c      if(mod(nevsk,100).eq.0) write(6,*) nrunsk,nevsk,nqisk,bsvertex(1),bsenergy,bsn50
c      if(nevsk.eq.100)  goto 9999

c      write(6,*) nrunsk,nevsk,nqisk,bsvertex(1),bsenergy,bsn50
      n=lecompte(px,py,pz,width)
      if(n .ne. 1)then
         goto 10
      end if

      rver = sqrt((px - bsvertex(1))**2 + (py - bsvertex(2))**2+ (pz - bsvertex(3))**2)
      if(rver.gt.300)then
         print *,'rver is big',rver,nevsk,nqisk, nevsk
         print *, px, bsvertex(1), py, bsvertex(2), pz, bsvertex(3)
         goto 10
      end if

c*** store lowe info.
      call skroot_set_lowe(lun,bsvertex, bsresult, bsdir, 
     $     bsgood, bsdirks, bseffhit, bsenergy, bsn50, bscossun,
     $     clvertex, clresult, cldir, clgoodness,     
     $     cldirks, cleffhit, clenergy, cln50, clcossun,
     $     latmnum, latmh, lmx24, ltimediff, lnsratio, lsdir, 
     $     spaevnum, spaloglike, sparesq, spadt, spadll, spadlt,
     $     spamuyn, spamugdn,
     $     posmc, dirmc, pabsmc, energymc, darkmc, islekeep,
     $     bspatlik, clpatlik, lwatert, lninfo, linfo)

      call skroot_set_tree(lun)
      call skroot_fill_tree(lun)

      goto 10
         
 1003 continue
      goto 10
 1002 continue
      write(0,*)'read error'
      goto 10
         
 1001 continue
         
c***  close files
      call skclosef(lun)
      call skroot_close(lun)
      call skroot_end

 9999 continue

c*** for bonsai 
      call cfbsexit

      stop
      end

c#include "lfallfit_sk4_gain_corr_mc.F"
#include "lecompte.F"
#include "linsum2.F"
