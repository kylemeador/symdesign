C COMPILE CMD: gfortran -o orient_oligomer_rmsd orient_oligomer_rmsd.f
C	orient_oligomer_rmsd.f - (special request by Dan) A general program for taking a PDB file containing
C		an oligomer with known point group symmetry and orienting that oligomer in some
C		canonical orientation, for example centered at the origin and with symmetry axes along
C		principle directions.
C
C	Limitations:
C		Will not work in any of the inifinite situations where a PDB file is fucked up,
C		in ways such as but not limited to:
C		equivalent residues in different chains don't have the same numbering; different subunits
C		are all listed with the same chain ID (e.g. with incremental residue numbering) instead
C		of separate IDs; multiple conformations are written out for the same subunit structure
C		(as in an NMR ensemble), negative residue numbers, etc. etc.
C
C		Assumes a homo-oligomer, as in A sub n, not AnBn.
C
C		Handles up to 60 subunits (i.e. symmetry I).
C
C		Handles up to 2000 residues in each chain.
C
C		Fails if the rotataional symmetry is broken by more than some allowed 'tolerance', for
C		example 2 degrees.
C
C		In case the input file contains chains whose residue ranges are not quite identical,
C		this program only writes out residues that are present in all chains.
C		Only the ATOM records are written out (and the occ and B are reset to 1 and 20).
C		Only the Calpha atoms are used for defining orientation mapping.
C
C		For defining the orientation between two sets of vectors, the program relies on old
C		routines from the IBM package for eigenvalues, and code written by TOY and DC Rees in
C		the '80s.
C		
C	Usage:
C		The input PDB is read from file 'input.pdb'
C		The output PDB is written to file 'output.pdb'
C
C		A short symmetry file containing specifications for the expected symmetry is read in from
C		a file whose name is requested from the user (i.e. read from standart input).  In the example
C		below for D4, that filename might be called D4_symm.txt
C
C		Example symmetry file for D4:
C
C			8
C			2
C			90., 180., 180.
C			0., 0., 1.
C			1. ,0., 0.
C			
C		Meaning:
C	      - 8 subunits in this symmetry
C	      - Mode = 2 (meaning symmetry higher than cyclic, which would be mode=1)
C	      - Pairwise angles: Three defining subunits in this symmetry (call them A, B, C,
C		which *do not* need to refer to chains A, B, and C in the PDB file) have pairwise
C		rotations between them of 90., 180., and 180.  That is, A to B is 90; A to C is 180;
C		B to C is 180.  In D4 this would describe the case of subunits A and B adjacent to each
C		other in one tetrameric ring, and subunit C in the other ring.
C	      - Components of a 3-D vector describing the direction you want the symmetry axis between
C		subunits A and B to point after the system is reoriented.
C	      - Components of a 3-D vector describing the direction you want the symmetry axis between
C		subunits A and C to point after the system is reoriented.
C		[so, this symmetry file will orient a D4 octamer so that the 4-fold is along z and the
C		2-folds are along x (and y and the x-y diagonals)]
C		Note that in order for this procedure to describe a unique orientation, the three
C		symmetry axes that relate A to B, A to C, and B to C, must be non-colinear.
C
C		example for C3 -
C			3
C			1
C			120.                (only need to specify rotation between 2 subunits)
C			0., 0., 1.
C	
C
C	T. O. Yeates - July 2014

C	modified to report rmsd overlap deviations - June 2020

C
	implicit none

	integer*4  i, j, k, nat, nch, ok(2000,60), comb(2000)
	real*4  xyz(3, 2000, 60), cm(3), rfinal(3,3), txfinal(3)
	integer*4  mode, nsymm, nres, nreslast
	real*4  symm_ang(3), outvec(3,2), allrot(3,3,60,60)
	real*4  allang(60,60), xyztmp(3), xyztmpout(3), rmsd_max
	character  junk*22


	call user_input (nsymm, mode, symm_ang, outvec)

	call get_coords (nch, xyz, comb)
	if (nch .ne. nsymm) stop 'Error: # subunits not equal to nsymm'

	call sub_center_of_mass (nch, xyz, comb, cm)

	call get_pairwise_rot_rmsd (nch,xyz,comb,allrot,allang,rmsd_max)
	write (6,*) 'Worst rmsd between pairs: ', rmsd_max

	if (mode .eq. 1) then
	  call subunit_mapping_m1
     %			(nch, allang, allrot, symm_ang, outvec, rfinal)
	  rewind (1)
	  do while (.true.)
 100	    read (1,901,err=100,end=200) junk, nres, xyztmp
	    if (junk(1:4) .eq. 'ATOM') then
	      if (comb(nres) .eq. 1) then
	        do j=1,3
	          xyztmp(j)=xyztmp(j)-cm(j)
	        end do
	        call mxmul (rfinal, xyztmp, xyztmpout, 3,3,1)
	        write (2,901) junk, nres, xyztmpout, 1., 20.
	      end if
	    end if
	  end do
 200	  continue
	end if
 901	format (a22, i4, 4x, 3f8.3, 2f6.2)

	if (mode .eq. 2) then
	  call subunit_mapping_m2
     %			(nch, allang, allrot, symm_ang, outvec, rfinal)
	  rewind (1)
	  do while (.true.)
 300	    read (1,901,err=300,end=400) junk, nres, xyztmp
	    if (junk(1:4) .eq. 'ATOM') then
	      if (comb(nres) .eq. 1) then
	        do j=1,3
	          xyztmp(j)=xyztmp(j)-cm(j)
	        end do
	        call mxmul (rfinal, xyztmp, xyztmpout, 3,3,1)
	        write (2,901) junk, nres, xyztmpout, 1., 20.
	      end if
	    end if
	  end do
 400	  continue
	end if

	stop
	end

C ===========================================================

	subroutine get_coords (nch, xyz, comb)

	implicit none

	integer*4  i, j, k, nat, nch, ok(2000,60), comb(2000), nres
	integer*4  nreslast
	real*4  xyz(3, 2000, 60), cm(3), rfinal(3,3), txfinal(3)
	real*4  xyztmp(3)
	character  junk*22, prevchain*1

	open (1, file='input.pdb', status='old')
	open (2, file='output.pdb', status='new')

	do i=1,2000
	  comb(i)=0
	  do j=1,60
	    ok(i,j)=0
	  end do
	end do

	nat=0
	nch=0
	prevchain='*'
	nreslast=-999
	do while (.true.)
 10	  read (1,901,err=10,end=20) junk, nres, xyztmp
	  if ((junk(1:4) .eq. 'ATOM' .or. (junk(1:4) .eq. 'HETA' .and.
     %	       junk(18:20) .eq. 'MSE')) .and. junk(14:15) .eq. 'CA') then
	   if (nres .ge. 1 .and. nres .le. 2000) then
	
	    if (junk(22:22) .ne. prevchain .or. nres .lt. nreslast) then
	
	      nch=nch+1
	      prevchain=junk(22:22)
	    end if

	    nreslast=nres

	    do i=1,3
	      xyz(i,nres,nch)=xyztmp(i)
	    end do
	    ok(nres,nch)=1
	   end if
	  end if
	end do
 20	continue
 901	format (a22, i4, 4x, 3f8.3)

	write (6,*) 'Number of chains = ', nch

	do i=1,2000
	  comb(i)=1
	  do j=1,nch
	    if (ok(i,j) .eq. 0) comb(i)=0
	  end do
C	  if (comb(i) .eq. 1) write (6,*) 'Fully present residue # = ', i
	end do

	return
	end

C ===========================================================

	subroutine sub_center_of_mass (nch, xyz, comb, cm)

	implicit none

	integer*4  i, j, k, nat, nch, ok(2000,60), comb(2000)
	real*4  xyz(3, 2000, 60), cm(3), rfinal(3,3), txfinal(3)
	integer*4  n

	do i=1,3
	  cm(i)=0.
	end do
	
	n=0
	do j=1,2000
	  if (comb(j) .eq. 1) then
	    do k=1,nch
	      n=n+1
	      do i=1,3
	        cm(i)=cm(i)+xyz(i,j,k)
	      end do
	    end do
	  end if
	end do

	do i=1,3
	  cm(i)=cm(i)/n
	end do

	write (6,*) ' Original center of mass = ', cm
	
	do j=1,2000
	  if (comb(j) .eq. 1) then
	    do k=1,nch
	      do i=1,3
	        xyz(i,j,k)=xyz(i,j,k)-cm(i)
	      end do
	    end do
	  end if
	end do

	return
	end

C ===========================================================

	subroutine user_input (nsymm, mode, symm_ang, outvec)

	implicit none

	integer*4  i, j, k
	integer*4  mode, nsymm
	real*4  symm_ang(3), outvec(3,2)
	character*60  fname

C	write (6,*) 'Input name of symmetry file'
C	read (5,901) fname
C	open (3,file=fname, status='old')
	open (3,file='symm.txt', status='old')

	read (3,*) nsymm
	read (3,*) mode
	if (mode .ne. 1 .and. mode .ne. 2) stop 'Error in value of MODE'
	if (mode .eq. 1) then
	  read (3,*) symm_ang(1)
	  read (3,*) (outvec(i,1),i=1,3)
	end if
	if (mode .eq. 2) then
	  read (3,*) symm_ang
	  read (3,*) ((outvec(i,j),i=1,3), j=1,2)
	end if
 901	format (a60)

	return
	end

C ===========================================================

	subroutine get_pairwise_rot_rmsd
     %			(nch, xyz, comb, allrot, allang, rmsd_max)

	integer*4  i, j, k, nat, nch, ok(2000,60), comb(2000), m
	real*4  xyz(3, 2000, 60), cm(3), rfinal(3,3), txfinal(3)
	integer*4  mode, nsymm
	real*4  symm_ang(3), outvec(3,2), allrot(3,3,60,60), allang(60,60)
	real*4  xtmp(3,10 000), ytmp(3,10 000), rotated(3,10 000)
	real*4  rot(3,3), tx(3), tr, angle


	nsymm=nch
	rmsd_max=0.

	do m=1,nsymm
  	  n=0
	  do i=1,2000
	    if (comb(i) .eq. 1) then
	      n=n+1
	      do j=1,3
	        xtmp(j,n)=xyz(j,i,m)
	      end do
	    end if
	  end do
	
	do k=1,nsymm
 	  n=0
	  do i=1,2000
	    if (comb(i) .eq. 1) then
	      n=n+1
	      do j=1,3
	        ytmp(j,n)=xyz(j,i,k)
	      end do
	    end if
	  end do
	
	  call rigid(n, xtmp, ytmp, rotated, rot, tx)

	  var=0.
	  do i=1,n
	    do j=1,3
	      var = var + (ytmp(j,i)-rotated(j,i))**2
	    end do
	  end do
	  rmsd=sqrt(var/n)

	  tr=0.
	  do i=1,3
	    tr=tr+rot(i,i)
	  end do
	  if (tr .ge. 3.) tr=3.
	  if (tr .le. -1.) tr=-1.
	  angle=180./3.1415926 * acos((tr-1.)/2.)
	  write (6,*) 'transformation: ', m, k, angle, tx, rmsd
	  if (rmsd .gt. rmsd_max) rmsd_max=rmsd
	  do i=1,3
	    do j=1,3
	      allrot(i,j,m,k)=rot(i,j)
	    end do
	  end do
	  allang(m,k)=angle

	end do
	end do

	return
	end

C ===========================================================

	subroutine subunit_mapping_m1
     %			(nch, allang, allrot, symm_ang, outvec, rfinal)


	integer*4  i, j, k, nat, nch, ok(2000,60), comb(2000), m
	real*4  xyz(3, 2000, 60), cm(3), rfinal(3,3), txfinal(3)
	integer*4  mode, nsymm
	real*4  symm_ang(3), outvec(3,2), allrot(3,3,60,60), allang(60,60)
	real*4  xtmp(3,10 000), ytmp(3,10 000), rotated(3,10 000)
	real*4  rot(3,3), tx(3), tr, angle, tolerance, axis(3), xyztmp(3)
	real*4  v1(3), v2(3), v3(3), rtmp1(3,3), rtmp2(3,3)
	real*4  w1(3), w2(3), w3(3)
	real*4  xyztmpout(3)
	character  junk*22

	tolerance=2.


	do i=2,nch
	  if (abs(symm_ang(1)-allang(1,i)) .le. tolerance) then
	    call get_axis(allrot(1,1,1,i), axis)
	    write (6,*) 'Axis = ', axis
C	    call get_rot_from_axis_m1(axis,outvec1,rfinal)
	    do j=1,3
	      v1(j)=outvec(j,1)
	      w1(j)=axis(j)
	    end do
	    call norm (v1, 3)
	    call norm (w1, 3)
	    call dotproduct (v1, w1, dotprod, 3)
	    if (abs(dotprod) .gt. (1-0.000005)) then
	      call set_ident (rfinal, 3)
	    else
	      call cross (v1, w1, v2)
	      call cross (v1, w1, w2)
	      call norm (v2, 3)
	      call norm (w2, 3)
	      call cross (v1, v2, v3)
	      call cross (w1, w2, w3)
	      do j=1,3
	        rtmp1(1,j)=w1(j)
	        rtmp1(2,j)=w2(j)
	        rtmp1(3,j)=w3(j)
	        rtmp2(j,1)=v1(j)
	        rtmp2(j,2)=v2(j)
	        rtmp2(j,3)=v3(j)
	      end do
	      call mxmul (rtmp2, rtmp1, rfinal, 3,3,3)
	    end if

	    goto 10
	  end if
	end do
	write (6,*)  'Error: Did not find a matching angle'
	stop
 10	continue

	return
	end

C ===========================================================

	subroutine subunit_mapping_m2
     %			(nch, allang, allrot, symm_ang, outvec, rfinal)


	integer*4  i, j, k, nat, nch, ok(2000,60), comb(2000), m
	real*4  xyz(3, 2000, 60), cm(3), rfinal(3,3), txfinal(3)
	integer*4  mode, nsymm
	real*4  symm_ang(3), outvec(3,2), allrot(3,3,60,60), allang(60,60)
	real*4  xtmp(3,10 000), ytmp(3,10 000), rotated(3,10 000)
	real*4  rot(3,3), tx(3), tr, angle, tolerance, axis(3), xyztmp(3)
	real*4  v1(3), v2(3), v3(3), rtmp1(3,3), rtmp2(3,3), axis2(3)
	real*4  w1(3), w2(3), w3(3), v4(3), w4(3)
	real*4  xyztmpout(3)
	character  junk*22

	tolerance=2.

	do i=2,nch
	  if (abs(symm_ang(1)-allang(1,i)) .le. tolerance) then
	    do k=2,nch
	      if (k .ne. i) then
	        if (abs(symm_ang(2)-allang(1,k)) .le. tolerance) then
	          if (abs(symm_ang(3)-allang(i,k)) .le. tolerance) then
	            
	    call get_axis(allrot(1,1,1,i), axis)
	    call get_axis(allrot(1,1,1,k), axis2)
	    call dotproduct (axis, axis2, dotprod, 3)
	    if (dotprod .lt. 0.) then
	      do j=1,3
	        axis2(j)=-axis2(j)
	      end do
	    end if
	    write (6,*) 'Axis = ', axis
	    write (6,*) 'Axis2 = ', axis2
	    do j=1,3
	      v1(j)=outvec(j,1)
	      v2(j)=outvec(j,2)
	      w1(j)=axis(j)
	      w2(j)=axis2(j)
	    end do
	    call dotproduct (v1, v2, dotprod, 3)
	    if (dotprod .lt. 0.) then
	      do j=1,3
	        v2(j)=-v2(j)
	      end do
	    end if
	    call norm (v1, 3)
	    call norm (v2, 3)
	    call norm (w1, 3)
	    call norm (w2, 3)

	    call cross (v1, v2, v3)
	    call norm (v3, 3)
	    call cross (v2, v3, v4)

	    call cross (w1, w2, w3)
	    call norm (w3, 3)
	    call cross (w2, w3, w4)

	    do j=1,3
	      rtmp1(1,j)=w2(j)
	      rtmp1(2,j)=w3(j)
	      rtmp1(3,j)=w4(j)
	      rtmp2(j,1)=v2(j)
	      rtmp2(j,2)=v3(j)
	      rtmp2(j,3)=v4(j)
	    end do
	    call mxmul (rtmp2, rtmp1, rfinal, 3,3,3)

	    goto 10

	end if
	end if
	end if
	end do

	  end if
	end do
	write (6,*)  'Error: Did not find a matching angle'
	stop
 10	continue

	return
	end

C ===========================================================

C	axis - determines the axis of rotation from a rotation matrix
C
	subroutine get_axis(r,rotaxis)

	implicit none

	real*4  a,b,c,d,e,f,g,h,i,det,t1,t2,t3,t4,t5,t6,t7,t8,t9
	real*4  inv1, inv2, inv3, inv4, x, y, dot
	real*4  rotaxis(3), r(3,3)
	integer*4  mode


C	read (5,*) a,b,c
C	read (5,*) d,e,f
C	read (5,*) g,h,i
	a=r(1,1)
	b=r(1,2)
	c=r(1,3)
	d=r(2,1)
	e=r(2,2)
	f=r(2,3)
	g=r(3,1)
	h=r(3,2)
	i=r(3,3)


	mode=0
	a=a-1.
	e=e-1.
	det=a*e-b*d
	if (abs(det) .lt. 0.0001) then
	  mode=1
	  a=a+1.
	  e=e+1.

	  t1=a
	  t2=b
	  t3=c
	  t4=d
	  t5=e
	  t6=f
	  t7=g
	  t8=h
	  t9=i

	  e=t1
	  f=t2
	  d=t3
	  h=t4
	  i=t5
	  g=t6
	  b=t7
	  c=t8
	  a=t9

	  a=a-1.
	  e=e-1.
	  det=a*e-b*d
	end if

	if (abs(det) .lt. 0.0001) then
	  mode=2
	  a=a+1.
	  e=e+1.

	  t1=a
	  t2=b
	  t3=c
	  t4=d
	  t5=e
	  t6=f
	  t7=g
	  t8=h
	  t9=i

	  e=t1
	  f=t2
	  d=t3
	  h=t4
	  i=t5
	  g=t6
	  b=t7
	  c=t8
	  a=t9

	  a=a-1.
	  e=e-1.
	  det=a*e-b*d
	end if

	inv1=e/det
	inv2=-b/det
	inv3=-d/det
	inv4=a/det

	x=inv1*c+inv2*f
	y=inv3*c+inv4*f
	x=-x
	y=-y

	dot=sqrt(1.+x*x+y*y)
	if (mode .eq. 0) then
	  rotaxis(1)=x/dot
	  rotaxis(2)=y/dot
	  rotaxis(3)=1./dot
	end if

	if (mode .eq. 1) then
	  rotaxis(3)=x/dot
	  rotaxis(1)=y/dot
	  rotaxis(2)=1./dot
	end if

	if (mode .eq. 2) then
	  rotaxis(2)=x/dot
	  rotaxis(3)=y/dot
	  rotaxis(1)=1./dot
	end if

C	write (6,*) rotaxis(1), rotaxis(2), rotaxis(3)

	return
	end


C ===========================================================


C
	subroutine  set_ident(a,n)

	integer*4  i, j, n
	real*4  a(n,n)

	do i=1,n
	  do j=1,n
	    a(i,j)=0.
	  end do
	  a(i,i)=1.
	end do

	return
	end

C
C ******************************************************
C	DOTPRODUCT - calculates dot product
C
	subroutine  dotproduct(a,b,d,n)

	integer*4  i, n
	real*4  a(1), b(1), d

	d=0.
	do i=1,n
	  d=d+a(i)*b(i)
	end do

	return
	end
C
C ******************************************************
C	NORM - normalizes a vector
C
	subroutine  norm(a,n)

	integer*4  n, i
	real*4  d, a(1)

	call dotproduct(a,a,d,n)
	if (d .le. 1.e-7) stop ' ERROR: vector length too small '
	d=sqrt(d)
	do i=1,n
	  a(i)=a(i)/d
	end do

	return
	end
C
C
C       CROSS - CALCULATES A CROSS-PRODUCT
C
        subroutine  cross(a,b,c)
C
        real*4  a(3), b(3), c(3)
C
        c(1) = a(2)*b(3) - a(3)*b(2)
        c(2) = -1.* (a(1)*b(3) - a(3)*b(1))
        c(3) = a(1)*b(2) - a(2)*b(1)
C
        return
        end

C ===========================================================


C     MXMUL.FOR
C
C     PERFORMS MATRIX MULTIPLICATION ON REAL MATRICES (A,B,&C)
C     C=A*B. A is IR by IRC. B is IRC by IC. C is IR by IC.
C
      SUBROUTINE MXMUL(A,B,C,IR,IRC,IC)
      DIMENSION A(1),B(1),C(1)
      DO 10 J=1,IC
      DO 10 I=1,IR
      IJ=IR*(J-1)+I
      C(IJ)=0.
      DO 10 K=1,IRC
      JK=IR*(K-1)+I
      KI=IRC*(J-1)+K
   10 C(IJ)=C(IJ)+A(JK)*B(KI)
      RETURN
      END
C ===========================================================


C *****************************************************************
C
C	rigid - 
C	given a model coordinate set x, with npt atoms and
C	a target set y, calculates the rotation matrix, rot,
C	and translation vector, t, to minimize
C        y  =  rot*x  +  d
C
C       yr is the actual rotated coordinates (ie.  rot*x+d)
C
C
C
C	D. C. Rees' implementation of Kabsch's algorithm.
C	T. O. Yeates - Bug fixed for degenerate case (npt =2)
C	
C	latest modification : 2-10-90
C
C
C
      subroutine rigid(npt,x,y,yr,rot,d)
      dimension y(3,10000),yr(3,10000),xp(3,10000),yp(3,10000)
      dimension x(3,npt),xcm(3),ycm(3)
      dimension r(3,3),rtr(3,3),b(3,3),eigv(3),eigvec(3,3)
      dimension rot(3,3),d(3)
      dimension temp(9),b1(3),b2(3),b3(3)
C
      if (npt .gt. 10000) stop ' FATAL ERROR : TOO MANY POINTS '
      if (npt .lt. 2) stop ' FATAL ERROR : TOO FEW POINTS '
      do 10 j=1,3
      xcm(j)=0.
      ycm(j)=0.
      do 15 i=1,npt
      xcm(j) = xcm(j) + x(j,i)
   15 ycm(j) = ycm(j) + y(j,i)
      xcm(j) = xcm(j)/float(npt)
   10 ycm(j) = ycm(j)/float(npt)
      do 20 i=1,npt
      do 20 j=1,3
      yp(j,i) = y(j,i) - ycm(j)
   20 xp(j,i) = x(j,i) - xcm(j)
C
C     set up R matrix and RTR
C
      do 30 i=1,3
      do 30 j=1,3
      r(i,j)=0.
      do 35 k=1,npt
   35 r(i,j) = r(i,j) + yp(i,k)*xp(j,k)
   30 continue
      do 40 i=1,3
      do 40 j=1,3
      rtr(i,j)=0.
      do 45 k=1,3
   45 rtr(i,j) = rtr(i,j) + r(k,i)*r(k,j)
   40 continue
C
C     now get eigenvalues and eigenvectors - use IBM's SSP package
C
      call mstr(rtr,temp,3,0,1)
      call eigen(temp,eigvec,3,0)
      call mstr(temp,eigv,3,1,2)
C
C     EIGEN returns eigenvalues sorted in descending fashion
C     set up a3 = a1 x a2
C
      if(eigv(3).le.0.00001) eigv(3) = 1.
      eigvec(1,3) = eigvec(2,1)*eigvec(3,2) - eigvec(2,2)*eigvec(3,1)
      eigvec(2,3) = eigvec(3,1)*eigvec(1,2) - eigvec(1,1)*eigvec(3,2)
      eigvec(3,3) = eigvec(1,1)*eigvec(2,2) - eigvec(2,1)*eigvec(1,2)
C
C     Now, get b vectors  = R ak / sqrt(eigval)
C
      do 50 k=1,3
      do 60 i=1,3
      b(i,k)=0.
      do 60 j=1,3
   60 b(i,k) = b(i,k) + r(i,j)*eigvec(j,k)/sqrt(eigv(k))
   50 continue
C
C     now, get b3 = b1 x b2
C
      dis1=0.
      dis2=0.
      do 76 i = 1,3
      b1(i) = b(i,1)
      dis1 = dis1 + b1(i)*b1(i)
      b2(i) = b(i,2)
      dis2 = dis2 + b2(i)*b2(i)
   76 b3(i) = b(i,3)
      dis1 = sqrt(dis1)
      dis2 = sqrt(dis2)
      do 75 i=1,3
      b1(i) = b1(i)/dis1
   75 b2(i) = b2(i)/dis2
      b3(1) = b1(2)*b2(3) - b2(2)*b1(3)
      b3(2) = b1(3)*b2(1) - b1(1)*b2(3)
      b3(3) = b1(1)*b2(2) - b1(2)*b2(1)
C      write (6,*) ' B vectors : '
C      do 64 i=1,3
C       write (6,*) (b(i,j),j=1,3)
C 64   continue
C
      do 81 i=1,3
       b(i,1)=b1(i)
       b(i,2)=b2(i)
       b(i,3)=b3(i)
 81   continue
C      write (6,*) ' Normalized B vectors '
C      do 82 i=1,3
C       write (6,*) (b(i,j),j=1,3)
C 82   continue
C
C     now, get u = b*a
C
      do 80 i = 1,3
      do 80 j = 1,3
      rot(i,j)=0.
      do 85 k = 1,3
   85 rot(i,j) = rot(i,j) + b(i,k)*eigvec(j,k)
   80 continue
C
C     translation vector
C
      do 90 i = 1,3
      d(i)=0.
      do 95 j = 1,3
   95 d(i) = d(i) - rot(i,j)*xcm(j)
   90 d(i) = d(i) + ycm(i)
C       write (6,*) ' Rotated coordinates : '
       do 200 in = 1,npt
       do 210 i = 1,3
        yr(i,in) = 0.
       do 220 j = 1,3
  220 yr(i,in)  = yr(i,in) + rot(i,j)*x(j,in)
  210 yr(i,in)  = yr(i,in) + d(i)
C       write(6,*) (yr(i,in),i=1,3)
  200 continue
      return
      end
C
C     ..................................................................
C
C        SUBROUTINE EIGEN
C
C        PURPOSE
C           COMPUTE EIGENVALUES AND EIGENVECTORS OF A REAL SYMMETRIC
C           MATRIX
C
C        USAGE
C           CALL EIGEN(A,R,N,MV)
C
C        DESCRIPTION OF PARAMETERS
C           A - ORIGINAL MATRIX (SYMMETRIC), DESTROYED IN COMPUTATION.
C               RESULTANT EIGENVALUES ARE DEVELOPED IN DIAGONAL OF
C               MATRIX A IN DESCENDING ORDER.
C           R - RESULTANT MATRIX OF EIGENVECTORS (STORED COLUMNWISE,
C               IN SAME SEQUENCE AS EIGENVALUES)
C           N - ORDER OF MATRICES A AND R
C           MV- INPUT CODE
C                   0   COMPUTE EIGENVALUES AND EIGENVECTORS
C                   1   COMPUTE EIGENVALUES ONLY (R NEED NOT BE
C                       DIMENSIONED BUT MUST STILL APPEAR IN CALLING
C                       SEQUENCE)
C
C        REMARKS
C           ORIGINAL MATRIX A MUST BE REAL SYMMETRIC (STORAGE MODE=1)
C           MATRIX A CANNOT BE IN THE SAME LOCATION AS MATRIX R
C
C        SUBROUTINES AND FUNCTION SUBPROGRAMS REQUIRED
C           NONE
C
C        METHOD
C           DIAGONALIZATION METHOD ORIGINATED BY JACOBI AND ADAPTED
C           BY VON NEUMANN FOR LARGE COMPUTERS AS FOUND IN 'MATHEMATICAL
C           METHODS FOR DIGITAL COMPUTERS', EDITED BY A. RALSTON AND
C           H.S. WILF, JOHN WILEY AND SONS, NEW YORK, 1962, CHAPTER 7
C
C     ..................................................................
C
      SUBROUTINE EIGEN(A,R,N,MV)
      REAL*4   A(1),R(1)
C
C        ...............................................................
C
C        IF A DOUBLE PRECISION VERSION OF THIS ROUTINE IS DESIRED, THE
C        C IN COLUMN 1 SHOULD BE REMOVED FROM THE DOUBLE PRECISION
C        STATEMENT WHICH FOLLOWS.
C
C      DOUBLE PRECISION A,R,ANORM,ANRMX,THR,X,Y,SINX,SINX2,COSX,
C     1                 COSX2,SINCS,RANGE
C
C        THE C MUST ALSO BE REMOVED FROM DOUBLE PRECISION STATEMENTS
C        APPEARING IN OTHER ROUTINES USED IN CONJUNCTION WITH THIS
C        ROUTINE.
C
C        THE DOUBLE PRECISION VERSION OF THIS SUBROUTINE MUST ALSO
C        CONTAIN DOUBLE PRECISION FORTRAN FUNCTIONS.  SQRT IN STATEMENTS
C        40, 68, 75, AND 78 MUST BE CHANGED TO DSQRT.  ABS IN STATEMENT
C        62 MUST BE CHANGED TO DABS. THE CONSTANT IN STATEMENT 5 SHOULD
C        BE CHANGED TO 1.0D-12.
C
C        ...............................................................
C
C        GENERATE IDENTITY MATRIX
C
    5 RANGE=1.0E-6
      IF(MV-1) 10,25,10
   10 IQ=-N
      DO 20 J=1,N
      IQ=IQ+N
      DO 20 I=1,N
      IJ=IQ+I
      R(IJ)=0.0
      IF(I-J) 20,15,20
   15 R(IJ)=1.0
   20 CONTINUE
C
C        COMPUTE INITIAL AND FINAL NORMS (ANORM AND ANORMX)
C
   25 ANORM=0.0
      DO 35 I=1,N
      DO 35 J=I,N
      IF(I-J) 30,35,30
   30 IA=I+(J*J-J)/2
      ANORM=ANORM+A(IA)*A(IA)
   35 CONTINUE
      IF(ANORM) 165,165,40
   40 ANORM=1.414*SQRT(ANORM)
      ANRMX=ANORM*RANGE/FLOAT(N)
C
C        INITIALIZE INDICATORS AND COMPUTE THRESHOLD, THR
C
      IND=0
      THR=ANORM
   45 THR=THR/FLOAT(N)
   50 L=1
   55 M=L+1
C
C        COMPUTE SIN AND COS
C
   60 MQ=(M*M-M)/2
      LQ=(L*L-L)/2
      LM=L+MQ
   62 IF(ABS(A(LM))-THR) 130,65,65
   65 IND=1
      LL=L+LQ
      MM=M+MQ
      X=0.5*(A(LL)-A(MM))
   68 Y=-A(LM)/SQRT(A(LM)*A(LM)+X*X)
      IF(X) 70,75,75
   70 Y=-Y
   75 SINX=Y/SQRT(2.0*(1.0+( SQRT(1.0-Y*Y))))
      SINX2=SINX*SINX
   78 COSX=SQRT(1.0-SINX2)
      COSX2=COSX*COSX
      SINCS =SINX*COSX
C
C        ROTATE L AND M COLUMNS
C
      ILQ=N*(L-1)
      IMQ=N*(M-1)
      DO 125 I=1,N
      IQ=(I*I-I)/2
      IF(I-L) 80,115,80
   80 IF(I-M) 85,115,90
   85 IM=I+MQ
      GO TO 95
   90 IM=M+IQ
   95 IF(I-L) 100,105,105
  100 IL=I+LQ
      GO TO 110
  105 IL=L+IQ
  110 X=A(IL)*COSX-A(IM)*SINX
      A(IM)=A(IL)*SINX+A(IM)*COSX
      A(IL)=X
  115 IF(MV-1) 120,125,120
  120 ILR=ILQ+I
      IMR=IMQ+I
      X=R(ILR)*COSX-R(IMR)*SINX
      R(IMR)=R(ILR)*SINX+R(IMR)*COSX
      R(ILR)=X
  125 CONTINUE
      X=2.0*A(LM)*SINCS
      Y=A(LL)*COSX2+A(MM)*SINX2-X
      X=A(LL)*SINX2+A(MM)*COSX2+X
      A(LM)=(A(LL)-A(MM))*SINCS+A(LM)*(COSX2-SINX2)
      A(LL)=Y
      A(MM)=X
C
C        TESTS FOR COMPLETION
C
C        TEST FOR M = LAST COLUMN
C
  130 IF(M-N) 135,140,135
  135 M=M+1
      GO TO 60
C
C        TEST FOR L = SECOND FROM LAST COLUMN
C
  140 IF(L-(N-1)) 145,150,145
  145 L=L+1
      GO TO 55
  150 IF(IND-1) 160,155,160
  155 IND=0
      GO TO 50
C
C        COMPARE THRESHOLD WITH FINAL NORM
C
  160 IF(THR-ANRMX) 165,165,45
C
C        SORT EIGENVALUES AND EIGENVECTORS
C
  165 IQ=-N
      DO 185 I=1,N
      IQ=IQ+N
      LL=I+(I*I-I)/2
      JQ=N*(I-2)
      DO 185 J=I,N
      JQ=JQ+N
      MM=J+(J*J-J)/2
      IF(A(LL)-A(MM)) 170,185,185
  170 X=A(LL)
      A(LL)=A(MM)
      A(MM)=X
      IF(MV-1) 175,185,175
  175 DO 180 K=1,N
      ILR=IQ+K
      IMR=JQ+K
      X=R(ILR)
      R(ILR)=R(IMR)
  180 R(IMR)=X
  185 CONTINUE
      RETURN
      END


C
C     ..................................................................
C
C        SUBROUTINE MSTR
C
C        PURPOSE
C           CHANGE STORAGE MODE OF A MATRIX
C
C        USAGE
C           CALL MSTR(A,R,N,MSA,MSR)
C
C        DESCRIPTION OF PARAMETERS
C           A - NAME OF INPUT MATRIX
C           R - NAME OF OUTPUT MATRIX
C           N - NUMBER OF ROWS AND COLUMNS IN A AND R
C           MSA - ONE DIGIT NUMBER FOR STORAGE MODE OF MATRIX A
C                  0 - GENERAL
C                  1 - SYMMETRIC
C                  2 - DIAGONAL
C           MSR - SAME AS MSA EXCEPT FOR MATRIX R
C
C        REMARKS
C           MATRIX R CANNOT BE IN THE SAME LOCATION AS MATRIX A
C           MATRIX A MUST BE A SQUARE MATRIX
C
C        SUBROUTINES AND FUNCTION SUBPROGRAMS REQUIRED
C           LOC
C
C        METHOD
C           MATRIX A IS RESTRUCTURED TO FORM MATRIX R.
C            MSA MSR
C             0   0  MATRIX A IS MOVED TO MATRIX R
C             0   1  THE UPPER TRIANGLE ELEMENTS OF A GENERAL MATRIX
C                    ARE USED TO FORM A SYMMETRIC MATRIX
C             0   2  THE DIAGONAL ELEMENTS OF A GENERAL MATRIX ARE USED
C                    TO FORM A DIAGONAL MATRIX
C             1   0  A SYMMETRIC MATRIX IS EXPANDED TO FORM A GENERAL
C                    MATRIX
C             1   1  MATRIX A IS MOVED TO MATRIX R
C             1   2  THE DIAGONAL ELEMENTS OF A SYMMETRIC MATRIX ARE
C                    USED TO FORM A DIAGONAL MATRIX
C             2   0  A DIAGONAL MATRIX IS EXPANDED BY INSERTING MISSING
C                    ZERO ELEMENTS TO FORM A GENERAL MATRIX
C             2   1  A DIAGONAL MATRIX IS EXPANDED BY INSERTING MISSING
C                    ZERO ELEMENTS TO FORM A SYMMETRIC MATRIX
C             2   2  MATRIX A IS MOVED TO MATRIX R
C
C     ..................................................................
C
      SUBROUTINE MSTR(A,R,N,MSA,MSR)
C      IMPLICIT DOUBLE PRECISION (A-H),(O-Z)
      real*4 A(1),R(1)
C
      DO 20 I=1,N
      DO 20 J=1,N
C
C        IF R IS GENERAL, FORM ELEMENT
C
      IF(MSR) 5,10,5
C
C        IF IN LOWER TRIANGLE OF SYMMETRIC OR DIAGONAL R, BYPASS
C
    5 IF(I-J) 10,10,20
   10 CALL LOCAT(I,J,IR,N,N,MSR)
C
C        IF IN UPPER AND OFF DIAGONAL  OF DIAGONAL R, BYPASS
C
      IF(IR) 20,20,15
C
C        OTHERWISE, FORM R(I,J)
C
   15 R(IR)=0.0
      CALL LOCAT(I,J,IA,N,N,MSA)
C
C        IF THERE IS NO A(I,J), LEAVE R(I,J) AT 0.0
C
      IF(IA) 20,20,18
   18 R(IR)=A(IA)
   20 CONTINUE
      RETURN
      END


C
C     ..................................................................
C
C        SUBROUTINE LOC
C
C        PURPOSE
C           COMPUTE A VECTOR SUBSCRIPT FOR AN ELEMENT IN A MATRIX OF
C           SPECIFIED STORAGE MODE
C
C        USAGE
C           CALL LOC (I,J,IR,N,M,MS)
C
C        DESCRIPTION OF PARAMETERS
C           I   - ROW NUMBER OF ELEMENT
C           J   - COLUMN NUMBER  OF ELEMENT
C           IR  - RESULTANT VECTOR SUBSCRIPT
C           N   - NUMBER OF ROWS IN MATRIX
C           M   - NUMBER OF COLUMNS IN MATRIX
C           MS  - ONE DIGIT NUMBER FOR STORAGE MODE OF MATRIX
C                  0 - GENERAL
C                  1 - SYMMETRIC
C                  2 - DIAGONAL
C
C        REMARKS
C           NONE
C
C        SUBROUTINES AND FUNCTION SUBPROGRAMS REQUIRED
C           NONE
C
C        METHOD
C           MS=0   SUBSCRIPT IS COMPUTED FOR A MATRIX WITH N*M ELEMENTS
C                  IN STORAGE (GENERAL MATRIX)
C           MS=1   SUBSCRIPT IS COMPUTED FOR A MATRIX WITH N*(N+1)/2 IN
C                  STORAGE (UPPER TRIANGLE OF SYMMETRIC MATRIX). IF
C                  ELEMENT IS IN LOWER TRIANGULAR PORTION, SUBSCRIPT IS
C                  CORRESPONDING ELEMENT IN UPPER TRIANGLE.
C           MS=2   SUBSCRIPT IS COMPUTED FOR A MATRIX WITH N ELEMENTS
C                  IN STORAGE (DIAGONAL ELEMENTS OF DIAGONAL MATRIX).
C                  IF ELEMENT IS NOT ON DIAGONAL (AND THEREFORE NOT IN
C                  STORAGE), IR IS SET TO ZERO.
C
C     ..................................................................
C
      SUBROUTINE LOCAT(I,J,IR,N,M,MS)
C
C      IMPLICIT DOUBLE PRECISION (A-H),(O-Z)
      IX=I
      JX=J
      IF(MS-1) 10,20,30
   10 IRX=N*(JX-1)+IX
      GO TO 36
   20 IF(IX-JX) 22,24,24
   22 IRX=IX+(JX*JX-JX)/2
      GO TO 36
   24 IRX=JX+(IX*IX-IX)/2
      GO TO 36
   30 IRX=0
      IF(IX-JX) 36,32,36
   32 IRX=IX
   36 IR=IRX
      RETURN
      END
