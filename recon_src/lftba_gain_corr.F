ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      subroutine lftba_gain_corr(elday,tba)
c
c	(purpos) calculate the top-bottom asymmetry
c	
c	(input) elday : elapse day  from 96/1/1
c
c	(output) top-bottom asymmetry from auto xenon
c
c	(date and author)	2009/5/28  Y.Koshio
c
ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc

      implicit none

#include "skhead.h"

      integer elday, extend, i, isave/0/, iday, ilast, itbaday(99999)
      real tba, tbavalue(99999), slope
      parameter (extend=10)
      character*200 file_name

      save isave, itbaday, ilast, tbavalue

      if(SK_GEOMETRY.le.2) then
         write(6,*) 'SK_GEOMETRY = ',SK_GEOMETRY
         write(6,*) 'This routine is only after SK-III'
         tba = 0.
         stop
      endif

      if(isave.eq.0) then
c         print*,'OPEN!'
         open(99,file='tba_gaincorr_20220826.dat',status='old')
         print *, 'TBA file: tba_gaincorr_20220826.dat'
c         open(99,file='tba_ax.dat',status='old')
c         call findconsts(file_name,'tba_ax','tba_ax.dat',2)
c         call findconsts(file_name,'tba_ax','tba_ax_gain.dat',2)
c         open(99,file=file_name,status='old')
         i=0
 10      read(99,*,end=11) iday, tba
         i=i+1
         itbaday(i) = iday
         tbavalue(i) = tba
         goto 10
 11      continue
         close(99)
         ilast = i
         isave = 1
      endif

******calculate water transparancy
      if(elday.lt.itbaday(1))then
         slope = (tbavalue(2)-tbavalue(1))/float(itbaday(2)-itbaday(1))
         tba = slope*(float(elday-itbaday(1)))+tbavalue(1)
         if(elday.le.(itbaday(1)-extend))then
            write(*,*)'Elapsed day (from 1/1/96) is ',elday
            write(*,*)
     $           'WARNING!  This is below the lower limit of current
     $           tob-bottom asymmetry table!!'
            write(*,*)'tob-bottom asymmetry may be incorrect!!!'
            write(*,*)' '
         end if
         return
      endif
      if(elday.ge.itbaday(ilast))then
         slope = (tbavalue(ilast)-tbavalue(ilast-1))/
     &        float(itbaday(ilast)-itbaday(ilast-1))
         tba = slope*(float(elday-itbaday(ilast)))
     &        +tbavalue(ilast)
         if(elday.ge.(itbaday(ilast)+extend))then
            write(*,*)'Elapsed day (from 1/1/96) is ',elday
            write(*,*)
     $           'WARNING!  This is above the upper limit of current
     $           tob-bottom asymmetry table!!'
            write(*,*)'Water transparancy value may be incorrect!!!'
            write(*,*)' '
         end if
         return
      end if

      do i=1,ilast-1
         if(elday.ge.itbaday(i).and.elday.lt.itbaday(i+1))then
            slope = (tbavalue(i+1)-tbavalue(i))/
     &           float(itbaday(i+1)-itbaday(i))
            tba = slope*(float(elday-itbaday(i)))+tbavalue(i)
	    return
         end if
      end do

      tba = 0.
      return
      end
