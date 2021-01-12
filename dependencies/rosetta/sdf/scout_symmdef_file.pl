#!/usr/bin/perl
##
##
###############################################################################

use strict;
use Math::Trig;	  # inv trig ops
use POSIX qw(ceil floor fmod fabs);
use List::Util qw(max min);
use File::Basename;
#use Getopt::Long qw(permute);
use constant PI	   => 4 * atan2(1, 1);

#use lib (".");
#use lib dirname(__FILE__);

###############################################################################

if ($#ARGV < 0) {
	print STDERR "usage: $0 [options]\n";
	print STDERR "\n";
	print STDERR "example:	 $0 -m NCS -a A -i B C -r 12.0 -p mystructure.pdb\n";
	print STDERR "\n";
	print STDERR "common options: \n";
	print STDERR "	  -m (NCS|PSEUDO) : [default NCS] which symmetric mode to run\n";
	print STDERR "			  NCS: generate noncrystallographic (point) symmetries from multiple chains in a PDB file\n";
	print STDERR "			  PSEUDO: (EXPERIMENTAL) generate pseudo-symmetric system\n";
	print STDERR "	  -p <string> : Input PDB file (one of -b or -p _must_ be given)\n";
	print STDERR "	  -r <real>	  : [default 1e6 for NCS+helix, 12.0 for cryst] the max CA-CA distance between two interacting chains\n";
	print STDERR "	  -q		  : [default false] quiet mode (no files are output)\n";
	print STDERR "\n";
	print STDERR "NCS-specific options: \n";
	print STDERR "	  -a <char>	  : [default A] the chain ID of the main chain\n";
	print STDERR "	  -d <char>*  : the chain ID of other chains to keep in the output file\n";
	print STDERR "	  -i <char>*  : [default B] the chain IDs of one chain in each symmetric subcomplex\n";
	print STDERR "	  -e		  : [default false] allow rigid body minimization of complete system\n";
	print STDERR "\n";
	print STDERR "PSEUDO-specific options: \n";
	print STDERR "	  -a <char>	  : [default A] the chain ID of the main chain\n";
	exit -1;
}


##	set default options
##
my $pdbfile = '';
my $interact_dist = 0.0;
my $primary_chain = 'A';
my %keep_chains = ();
my @secondary_chains = ();
my $helical_chain = 'B';
my $rbminAll = 0;
my @cell_new;
my @cell_offset;
my $spacegp_new;
my $modestring = "NCS";
my $nturns = 4;
my $quietMode = 0;
my $restrictCrystTrans = 0;


## parse options (do this by hand since Getopt does not handle this well)
##
my $inlinefull = (join ' ',@ARGV)." ";
my @suboptions = split /(-[a-z|A-Z] )/, $inlinefull;
for ( my $i=0; $i<=$#suboptions; $i++ ) {
	if ($suboptions[$i] eq "-m " && defined $suboptions[$i+1] && $suboptions[$i+1] !~ /^-[a-z|A-Z]/) {
		$modestring = $suboptions[++$i];
		$modestring =~ s/\s*(\S+)\s*/$1/;
	}
	elsif ($suboptions[$i] eq "-p " && defined $suboptions[$i+1] && $suboptions[$i+1] !~ /^-[a-z|A-Z]/) {
		$pdbfile = $suboptions[++$i];
		$pdbfile =~ s/\s*(\S+)\s*/$1/;
	}
	elsif ($suboptions[$i] eq "-r " && defined $suboptions[$i+1] && $suboptions[$i+1] !~ /^-[a-z|A-Z]/) {
		$interact_dist = $suboptions[++$i];
	}
	elsif ($suboptions[$i] =~ /^-q/ ) {
		$quietMode = 1;
	}
	elsif ($suboptions[$i] =~ /^-e/ ) {
		$rbminAll = 1;
	}
	elsif ($suboptions[$i] =~ /^-h/ ) {
		$restrictCrystTrans = 1;
	}
	# cryst
	elsif ($suboptions[$i] eq "-c " && defined $suboptions[$i+1] && $suboptions[$i+1] !~ /^-[a-z|A-Z]/) {
		@cell_new = split /[, ]/,$suboptions[++$i];
	}
	elsif ($suboptions[$i] eq "-g " && defined $suboptions[$i+1] && $suboptions[$i+1] !~ /^-[a-z|A-Z]/) {
		@cell_offset = split /[, ]/,$suboptions[++$i];
	}
	elsif ($suboptions[$i] eq "-s " && defined $suboptions[$i+1] && $suboptions[$i+1] !~ /^-[a-z|A-Z]/) {
		$spacegp_new = $suboptions[++$i];
	} elsif ($suboptions[$i] eq "-t " && defined $suboptions[$i+1] && $suboptions[$i+1] !~ /^-[a-z|A-Z]/) {
		$nturns = ( $suboptions[++$i] );
	} elsif ($suboptions[$i] eq "-a " && defined $suboptions[$i+1] && $suboptions[$i+1] !~ /^-[a-z|A-Z]/) {
		$primary_chain = $suboptions[++$i];
		$primary_chain =~ s/\s*(\S+)\s*/$1/;
	} elsif ($suboptions[$i] eq "-b " && defined $suboptions[$i+1] && $suboptions[$i+1] !~ /^-[a-z|A-Z]/) {
		$helical_chain = $suboptions[++$i];
		$helical_chain =~ s/\s*(\S+)\s*/$1/;
	} elsif ($suboptions[$i] eq "-i " && defined $suboptions[$i+1] && $suboptions[$i+1] !~ /^-[a-z|A-Z]/) {
		@secondary_chains = split /[, ]/,$suboptions[++$i];
	} elsif ($suboptions[$i] eq "-d " && defined $suboptions[$i+1] && $suboptions[$i+1] !~ /^-[a-z|A-Z]/) {
		my @keep_chain_list = split /[, ]/,$suboptions[++$i];
		foreach my $keepchain ( @keep_chain_list ) {
			$keep_chains{ $keepchain } = 1;
		}
	} elsif (length($suboptions[$i]) > 0) {
		print STDERR "Error parsing command line while at '".$suboptions[$i]."'\n";
	}
}

##
if ($pdbfile eq '') {
	print STDERR "Must provide an input PDB file with -p\n";
	exit -1;
}

## parse mode string
##	  + set mode-specific defaults
##
my ($ncs_mode, $pseudo_mode) = (0,0,0,0);  # $cryst_mode, $helix_mode,
# if ($modestring eq "CRYST" || $modestring eq "cryst" || $modestring eq "Cryst") {
# 	$cryst_mode = 1;
# } els
if ($modestring eq "NCS" || $modestring eq "ncs" || $modestring eq "Ncs") {
	$ncs_mode = 1;
	if ( scalar(@secondary_chains) == 0) {
		@secondary_chains = ('B');
	}
# } elsif ($modestring eq "HELIX" || $modestring eq "helix" || $modestring eq "Helix") {
# 	$helix_mode = 1;
} elsif ($modestring eq "PSEUDO" || $modestring eq "pseudo" || $modestring eq "Pseudo") {
	$pseudo_mode = 1;
} else {
	print STDERR "Unrecognized mode string '$modestring'\n";
	exit -1;
}
# if ($quietMode!= 1) { print STDERR "Running in mode $modestring.\n"; }

## set default interaction radius (mode-specific)
# if ($interact_dist == 0) {
# 	if ($cryst_mode) {
# 		$interact_dist = 12.0;
# 	} else {
# 		$interact_dist = 1e6;
# 	}
# }

## substitute'_' -> ' '
if ($primary_chain eq '_') {
	$primary_chain = ' ';
}
if ($helical_chain eq '_') {
	$primary_chain = ' ';
}
foreach my $i (0..$#secondary_chains) {
	if ($secondary_chains[$i] eq '_') {
		$secondary_chains[$i] = ' ';
	}
}


###
### Read input PDB file
my %chains;		 # input as separate chains
my @chaintrace;	 # input as a single monomer
my @filebuf;
my @altfilebuf;
my $minRes = 1;
my $monomerRadius = 0;

# crystinfo
my $spacegp = "P 1";
my ($gpid, $nsymm, $Rs, $Ts, $Cs, $cheshire);
my ($A, $B, $C, $alpha, $beta, $gamma) = (0,0,0,90,90,90);
my ($f2c,$c2f);

my $CoM = [0,0,0];
my $anchor_ca = [0,0,0];
my $CoM;


if ($pdbfile =~ "\.gz\$") {
	open (PDB, "gunzip -c $pdbfile | ") || die "Cannot open $pdbfile.";
} else {
	open (PDB, $pdbfile) || die "Cannot open $pdbfile.";
}
while (<PDB>) {
	chomp;
	if (/^ATOM/ || /^HETATM/) {
		my $chnid = substr ($_, 21, 1);
		my $atom  = substr ($_, 12, 4);
		if ($atom eq " CA ") {
			if (!defined $chains{ $chnid } ) {
				$chains{ $chnid } = [];
			}
			my $CA_i = [substr ($_, 30, 8),substr ($_, 38, 8),substr ($_, 46, 8)];
			push @{ $chains{ $chnid } }, $CA_i;
		}
		if ($primary_chain eq $chnid) {
			push @filebuf, $_;
		} elsif ( defined $keep_chains{ $chnid } ) {
			push @altfilebuf, $_;
		}
	}
}
close (PDB);


## recenter the primary chain
##
if ( ! defined $chains{ $primary_chain } ) {
	die "Chain '$primary_chain' not in input!\n";
}
## find residue closest to CoM of the system
## find the radius of the molecule (for -f)
my $maxDist2 = 0;
my $minDist2 = 9999999;
foreach my $i ( 0..scalar( @{ $chains{ $primary_chain } })-1 ) {
	my $dist2 = vnorm2( $chains{ $primary_chain }->[$i] );
	if ($dist2 < $minDist2) {
		$minDist2 = $dist2;
		$minRes = $i+1;
		$anchor_ca = deep_copy( $chains{ $primary_chain }->[$i] );
	}
	if ($dist2 > $maxDist2) {
		$maxDist2 = $dist2;
	}
}
$monomerRadius = sqrt( $maxDist2 );
# }


####################################################################################
####################################################################################
###	  mode-specific stuff
####################################################################################
####################################################################################

if ($ncs_mode == 1) {
	my $NCS_ops = {};
	my $R_0 = [ [1,0,0], [0,1,0], [0,0,1] ];  # R_0
	my $COM_0 = recenter( $chains{ $primary_chain } );
	$NCS_ops->{R} = $R_0;
	$NCS_ops->{T} = $COM_0;
	$NCS_ops->{PATH} = "";
	$NCS_ops->{CHILDREN} = [];
	$NCS_ops->{AXIS} = [0,0,1];


	my @allQs;
	my @allCOMs;
	my @sym_orders;

	foreach my $sec_chain (@secondary_chains) {
		my @sec_chain_ids = split( ':', $sec_chain );

		if ( ! defined $chains{ $sec_chain_ids[0] } ) {
			die "Chain $sec_chain not in input!\n";
		}

		## CA check
		# if (scalar( @{ $chains{ $primary_chain } } ) != scalar( @{ $chains{ $sec_chain_ids[0] } } ) ) {
		# 	print STDERR "ERROR! chains '$primary_chain' and '$sec_chain' have different residue counts! (".
		# 				 scalar( @{ $chains{ $primary_chain } } )." vs ".scalar( @{ $chains{ $sec_chain_ids[0] } } ).")\n";
		# 	die "Chain length mismatch!\n";
		# }

		# get superposition
		my ($R,$rmsd, $COM_i, $COM_ij) = rms_align( $chains{ $primary_chain } , $chains{ $sec_chain_ids[0] } );
		# if ($quietMode != 1) {
			# print STDERR "Aligning $primary_chain and $sec_chain wth RMS=$rmsd.\n";
		# }

		# if ( is_identity( $R ) ) {
			# print STDERR "Chains $primary_chain and $sec_chain related by transformation only! Aborting.\n";
			# exit 1;
		# }

		my $del_COM = vsub ($COM_i, $COM_0);
		push @allCOMs, $del_COM;

		my ($X,$Y,$Z,$W)=R2quat($R);

		my $Worig = $W;
		my $Wmult = 1;
		if ($W < 0) { $W = -$W; $Wmult = -1; }
		my $omega = acos($W);
		my $sym_order = int(PI/$omega + 0.5);

		# optionally allow input to 'force' a symmetric order
		if ($#sec_chain_ids > 0) {
			$sym_order = $sec_chain_ids[1];
		}
		push @sym_orders, $sym_order;

		my $rotaxis = [$X,$Y,$Z]; normalize( $rotaxis );
		# if ($quietMode != 1) {
		print "".$sec_chain_ids[0].":".$sym_order."-fold axis";
	    # print ": ".$rotaxis->[0]." ".$rotaxis->[1]." ".$rotaxis->[2]."\n";
	    printf(": %10.8f %10.8f %10.8f\n", $rotaxis->[0], $rotaxis->[1], $rotaxis->[2]);
	    	# }

		# now make perfectly symmetrical version of superposition
		my $newW = -$Wmult *cos( PI/$sym_order );
		my $S = sqrt ( (1-$newW*$newW)/($X*$X+$Y*$Y+$Z*$Z) );
		my $newQ = [$X*$S , $Y*$S, $Z*$S, $newW];
		push @allQs, $newQ;
	}
}


if ($pseudo_mode == 1) {
	###############
	### PSEUDO mode!
	###############
	my $R_0 = [ [1,0,0], [0,1,0], [0,0,1] ];  # R_0
	my $COM_0 = recenter( $chains{ $primary_chain } );

	my @Rs;
	my @Ts;
	push @Rs, $R_0;
	push @Ts, $COM_0;


	foreach my $sec_chain (sort keys %chains) {
		next if ($sec_chain eq $primary_chain);
		print STDERR "Chain $sec_chain -> $primary_chain\n";

		# superpose, steal R and T
		my ($R,$rmsd, $COM_i, $COM_ij) = rms_align( $chains{ $primary_chain } , $chains{ $sec_chain } );
		my $del_COM = vsub ($COM_i, $COM_0);
		my $Rinv = minv( $R );
		push @Rs, $Rinv;
		push @Ts, $COM_i;
	}


	#######################################
	##
	## write output symm file
	# symmetry_name c4
	# subunits 4
	# number_of_interfaces 2
	# E = 3*E2
	# anchor_residue 17
	my $symmname = $pdbfile;
	$symmname =~ s/\.pdb$//;
	$symmname = $symmname."_pseudo".scalar(keys %chains)."fold";
	print "symmetry_name $symmname\n";

	print "E = 2*VRT_0_base";
	foreach my $i (1..($#Rs)) {
		my $estring = " + 1*(VRT_0_base:VRT_".($i)."_base)";
		$estring =~ s/_-(\d)/_n\1/g;
		print $estring;
	}
	print "\n";
	print "anchor_residue COM\n";

	# virtual_coordinates_start
	# xyz VRT1 -1,0,0 0,1,0 0,0,0
	# xyz VRT2 0,-1,0 -1,0,0 0,0,0
	# xyz VRT3 1,0,0 0,-1,0 0,0,0
	# xyz VRT4 0,1,0 1,0,0 0,0,0
	# virtual_coordinates_stop
	print "virtual_coordinates_start\n";
	foreach my $i (0..($#Rs)) {
		# crystal lattice
		my $xyzline = "xyz VRT_".$i;
		$xyzline =~ s/_-(\d)/_n\1/g;

		# X
		my $rX	= mapply( $Rs[$i] , [1,0,0] );
		my $string = sprintf("%.6f,%.6f,%.6f", $rX->[0], $rX->[1], $rX->[2]);
		$xyzline = $xyzline." ".$string;

		# Y
		my $rY	= mapply( $Rs[$i] , [0,1,0] );
		$string = sprintf("%.6f,%.6f,%.6f", $rY->[0], $rY->[1], $rY->[2]);
		$xyzline = $xyzline." ".$string;

		# orig
		my $ori = $Ts[$i];
		$string = sprintf("%.6f,%.6f,%.6f", $ori->[0], $ori->[1], $ori->[2]);
		$xyzline = $xyzline." ".$string;
		print "$xyzline\n";

		# centers of mass
		# X
		my $xyzline = "xyz VRT_".$i."_base";
		$xyzline =~ s/_-(\d)/_n\1/g;
		my $string = sprintf("%.6f,%.6f,%.6f", $rX->[0], $rX->[1], $rX->[2]);
		$xyzline = $xyzline." ".$string;

		# Y
		$string = sprintf("%.6f,%.6f,%.6f", $rY->[0], $rY->[1], $rY->[2]);
		$xyzline = $xyzline." ".$string;

		# orig
		my $ori = $Ts[$i];
		$string = sprintf("%.6f,%.6f,%.6f", $ori->[0], $ori->[1], $ori->[2]);
		$xyzline = $xyzline." ".$string;
		print "$xyzline\n";
	}
	print "virtual_coordinates_stop\n";

	# connect_virtual BASEJUMP VRT1 SUBUNIT
	# connect_virtual JUMP2 VRT2 SUBUNIT
	# connect_virtual JUMP10 VRT1 VRT2
	#print "connect_virtual BASEJUMP VRT_0_0_0_0 SUBUNIT\n";  # t_0
	#jump from com to subunit
	foreach my $i (0..($#Rs)) {
		print "connect_virtual JUMP_".$i."_to_subunit VRT_".$i."_base SUBUNIT\n";
	}
	#jump to com
	foreach my $i (0..($#Rs)) {
		print "connect_virtual JUMP_".$i."_to_com VRT_".$i." VRT_".$i."_base\n";
	}
	#jump from base unit
	foreach my $i (1..($#Rs)) {
		print "connect_virtual JUMP_".$i." VRT_0 VRT_".$i."\n";
	}
	print "set_dof JUMP_0_to_com x y z\n";
	print "set_dof JUMP_0_to_subunit angle_x angle_y angle_z\n";

	#define jumpgroups
	print "set_jump_group JUMPGROUP1 ";
	foreach my $i (0..($#Rs)) {
		print " JUMP_".$i."_to_subunit";
	}
	print "\n";
	print "set_jump_group JUMPGROUP2 ";
	foreach my $i (0..($#Rs)) {
		print " JUMP_".$i."_to_com";
	}
	print "\n";


	########################################
	##
	## write output pdb
	my $outpdb = $pdbfile;
	my $outmon = $pdbfile;
	if ($outpdb =~ /\.pdb$/) {
		$outpdb =~ s/\.pdb$/_symm.pdb/;
		$outmon =~ s/\.pdb$/_INPUT.pdb/;
	} else {
		$outpdb = $outpdb."_symm.pdb";
		$outmon = $outpdb."_INPUT.pdb";
	}
	open (OUTPDB, ">$outpdb");
	open (OUTMON, ">$outmon");
	my $chnidx = 0;
	my $chains = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz1234567890!@#$%^&*()-=_+;:,.<>";
	foreach my $i (0..($#Rs)) {
		foreach my $line (@filebuf) {
			my $linecopy = $line;

			my $X = vsub([substr ($line, 30, 8),substr ($line, 38, 8),substr ($line, 46, 8)] , $COM_0);
			my $rX = vadd( mapply($Rs[$i],$X) , $Ts[$i] );

			substr ($linecopy, 30, 8) = sprintf ("%8.3f", $rX->[0]);
			substr ($linecopy, 38, 8) = sprintf ("%8.3f", $rX->[1]);
			substr ($linecopy, 46, 8) = sprintf ("%8.3f", $rX->[2]);
			substr ($linecopy, 21, 1) = substr ($chains, $chnidx, 1);

			print OUTPDB $linecopy."\n";
		}
		print OUTPDB "TER	\n";
		$chnidx++;
	}
	close(OUTPDB);

	######################################

	foreach my $line (@filebuf) {
		my $linecopy = $line;

		my $X = [substr ($line, 30, 8),substr ($line, 38, 8),substr ($line, 46, 8)];

		substr ($linecopy, 30, 8) = sprintf ("%8.3f", $X->[0]);
		substr ($linecopy, 38, 8) = sprintf ("%8.3f", $X->[1]);
		substr ($linecopy, 46, 8) = sprintf ("%8.3f", $X->[2]);
		substr ($linecopy, 21, 1) = "A";

		print OUTMON $linecopy."\n";
	}

}


#############################################################################
#############################################################################
###
###	 NCS symmetry tree generation fns.
###
#############################################################################
#############################################################################

# expand_symmops_by_split( $NCS_ops, $newR, $adj_newDelCOM, $sym_order)
sub expand_symmops_by_split {
	my ( $tree, $newR, $newDelT, $sym_order, $axis_i ) = @_;

	my $newNCSops = {};
	my $COM_0 = [0 , 0 , 0];
	$newNCSops->{R} = [ [1,0,0], [0,1,0], [0,0,1] ];
	$newNCSops->{CHILDREN} = [];
	$newNCSops->{AXIS} = deep_copy($axis_i); #axis of children

	my $COM_i = [ 0,0,0 ];
	my $R_i	  = [ [1,0,0], [0,1,0], [0,0,1] ];
	my $newCOM0 = [0,0,0];

	foreach my $i (0..$sym_order-1) {
		#my $newNCSops_i = deep_copy( $NCS_ops );
		my $newNCSops_i = deep_copy( $tree );

		# rotate about the center of mass of the subtree
		#	 then translate to the new CoM
		#apply_transformation ( $newNCSops_i, $R_i, $NCS_ops->{T}, $COM_i, $i );
		apply_transformation ( $newNCSops_i, $R_i, $tree->{T}, $COM_i, $i );
		$newNCSops_i->{AXIS} = deep_copy($axis_i);
		push @{ $newNCSops->{CHILDREN} }, $newNCSops_i;

		$newCOM0 = vadd( $newCOM0, $newNCSops_i->{T} );

		$COM_i = vadd( $COM_i , mapply( $R_i, $newDelT ) );
		$R_i = mmult( $newR , $R_i );
	}
	$newNCSops->{T} = [ $newCOM0->[0]/$sym_order , $newCOM0->[1]/$sym_order , $newCOM0->[2]/$sym_order ];
	$newNCSops->{PATH} = "";
	$_[0] = $newNCSops;
}


#############
# apply a transform ... rotate about $T_about, applying a post_transform $T_post
sub apply_transformation {
	my ($tree, $R, $T_about, $T_post, $prefix) = @_;

	$tree->{R} = mmult( $R, $tree->{R} );
	$tree->{T} = vadd( vadd($T_about, $T_post) , mapply( $R, vsub( $tree->{T} , $T_about ) ) );
	my $newPath = $prefix;
	if (length($tree->{PATH}) > 0) { $newPath = $newPath.'_'.$tree->{PATH}; }
	$tree->{PATH} = $newPath;
	foreach my $child ( @{ $tree->{CHILDREN} } ) {
		apply_transformation( $child, $R, $T_about, $T_post, $prefix);
	}
}


##########
## ($nnodes,$nleaves) = tree_size( $tree )
##	  get the size of a tree
sub tree_size {
	my $tree = shift;
	my ($nnodes,$nleaves) = (0,0);
	tree_size_recursive( $tree, $nnodes,$nleaves);
	return ($nnodes,$nleaves);
}

sub tree_size_recursive {
	my ($tree, $nnodes, $nleaves) = @_;
	my $nchildren = scalar( @{ $tree->{CHILDREN} } );

	$nnodes++;
	if ($nchildren == 0) {
		$nleaves++;
	} else {
		foreach my $child ( @{ $tree->{CHILDREN} } ) {
			tree_size_recursive( $child, $nnodes, $nleaves);
		}
	}

	# pass-by-ref
	@_[1] = $nnodes;
	@_[2] = $nleaves;
}


############
## my $leaves = tree_traverse( $tree)
##	 traverse the tree
sub tree_traverse {
	my $tree = shift;
	my $leaves = [];
	tree_traverse_recursive( $tree, $leaves );
	return $leaves;
}

sub tree_traverse_recursive {
	my ($tree,$leaves) = @_;
	my $nchildren = scalar( @{ $tree->{CHILDREN} } );
	if ($nchildren == 0) {
		push @{ $leaves }, $tree;
	} else {
		foreach my $child ( @{ $tree->{CHILDREN} } ) {
			tree_traverse_recursive( $child, $leaves );
		}
	}
}

############
## my $vrts_by_depth = tree_traverse_by_depth( $NCS_ops );
##	   traverse every node in the tree returning them sorted by depth
sub tree_traverse_by_depth {
	my $tree = shift;
	my $depth = get_depth( $tree );
	my $nodes_by_depth = [	];
	tree_traverse_by_depth_recursive( $tree, $nodes_by_depth, 0 );
	return $nodes_by_depth;
}

sub tree_traverse_by_depth_recursive {
	my ( $tree, $nodes_by_depth , $curr_depth) = @_;
	my $nchildren = scalar( @{ $tree->{CHILDREN} } );

	if (!defined $nodes_by_depth->[$curr_depth] ) {
		$nodes_by_depth->[$curr_depth] = [];
	}
	push @{ $nodes_by_depth->[$curr_depth] }, $tree->{PATH};

	foreach my $child ( @{ $tree->{CHILDREN} } ) {
		tree_traverse_by_depth_recursive( $child, $nodes_by_depth , $curr_depth+1 );
	}
}


##########
## get a string describing the topology of the tree
sub get_topology {
	my $tree = shift;
	my $nchildren = scalar( @{ $tree->{CHILDREN} } );
	my $retval = "";
	if ($nchildren != 0) {
		# just look at first child
		$retval = $retval."_$nchildren".get_topology( $tree->{CHILDREN}->[0] );
	}
	return $retval;
}

##########
## get the depth
sub get_depth {
	my $tree = shift;
	my $nchildren = scalar( @{ $tree->{CHILDREN} } );
	my $retval = 0;
	if ($nchildren != 0) {
		$retval = 1+get_depth( $tree->{CHILDREN}->[0] );
	}
	return $retval;
}

#########
## access a subtree
sub get_subtree {
	my ($tree,$accessor_string) = @_;
	my @accessor_list = split '_',$accessor_string;
	#shift @accessor_list; # pop initial '0' off accessor list
	my $subtree_j = get_subtree_recursive( $tree, \@accessor_list );
}

sub get_subtree_recursive {
	my $tree = shift;
	my $list = shift;
	if (scalar( @{ $list } ) == 0 || $list->[0] eq "" ) {
		return $tree;
	} else {
		my $idx = shift @{ $list };
		return get_subtree_recursive( $tree->{CHILDREN}->[$idx] , $list );
	}
	print "ERROR!  Subtree undefined\n";
	exit -1 ;
}


#########
## ($vrt_lines, $connect_lines, $dof_lines) = fold_tree_from_ncs( $NCS_ops );
##
## This function recreates an NCS coordinate system from an NCS tree
sub fold_tree_from_ncs {
	my ($tree, $nodes_by_depth, $connected_subunits ) = @_;

	# root doesnt have parents or siblings
	# use 1st child instead
	my $axis = $tree->{AXIS};

	# parent com (arbitrarily) perpendicular to rot axis
	my $paxis;
	if ($axis->[1] != 0 || $axis->[2] != 0) {
		$paxis = cross([1,0,0], $axis);
	} else {
		$paxis = cross([0,1,0], $axis);
	}
	my $parent_com = vadd( $paxis ,get_com( $tree ) );

	my $nsiblings = 1;
	my $vrt_lines = [];
	my $connect_lines = [];
	my $dof_lines = [];

	fold_tree_from_ncs_recursive( $tree , $parent_com, $nsiblings, $vrt_lines,
								  $connect_lines, $dof_lines, $nodes_by_depth, $connected_subunits );
	return ( $vrt_lines, $connect_lines, $dof_lines);
}


sub fold_tree_from_ncs_recursive {
	my ($tree,$parent_com,$nsiblings,
		$vrt_lines,$connect_lines,$dof_lines,$nodes_by_depth,$connected_subunits) = @_;

	my $origin = get_com( $tree );
	my $id_string = $tree->{PATH};

	# x points from origin to parent CoM
	my $myX = vsub( $parent_com , $origin );
	my $myZ = $tree->{AXIS}; #mapply( $tree->{R}, [0,0,1]);

	# y is whatever is left
	my $myY = cross( $myZ, $myX );

	# at top level, if symm axis is X, this can happen
	if (abs($myY->[0]) < 1e-6 && abs($myY->[1]) < 1e-6 && abs($myY->[2]) < 1e-6 ) {
		$myZ = mapply( $tree->{R}, [1,0,0]);
		$myY = cross( $myZ, $myX );
	}
	normalize( $myX );
	normalize( $myY );

	#
	my $nchildren = scalar( @{ $tree->{CHILDREN} } );

	## recursive call
	foreach my $child_idx ( 0..$nchildren-1 ) {
		my $child_idstring = $tree->{CHILDREN}->[$child_idx]->{PATH};

		# now recursively call each child
		fold_tree_from_ncs_recursive( $tree->{CHILDREN}->[$child_idx], $origin, $nchildren ,
									  $vrt_lines, $connect_lines, $dof_lines, $nodes_by_depth,$connected_subunits);
	}

	# xyz VRT1 -1,0,0 0,1,0 0,0,0
	push @{ $vrt_lines }, "xyz VRT$id_string  ".
				sprintf("%.7f,%.7f,%.7f", $myX->[0], $myX->[1], $myX->[2])."  ".
				sprintf("%.7f,%.7f,%.7f", $myY->[0], $myY->[1], $myY->[2])."  ".
				sprintf("%.7f,%.7f,%.7f", $parent_com->[0], $parent_com->[1], $parent_com->[2]);

	## set up vrts + jumps
	if ($nchildren > 0) {
		## jump to first child
		my $child0_idstring = $tree->{CHILDREN}->[0]->{PATH};
		push @{ $connect_lines }, "connect_virtual JUMP$child0_idstring VRT$id_string VRT$child0_idstring";

		# is this jump the controlling jump?
		my $is_controlling = -1;
		my $depth = 0;
		foreach my $nodes_i (@{ $nodes_by_depth }) {
			if ($child0_idstring eq $nodes_i->[0]) {
				$is_controlling = $depth;
			}
			$depth++;
		}

		if ($is_controlling >= 1) {	 # level 1 jumps don't move, hence '>'
			my $x_dist = vnorm( vsub( $parent_com,$origin) ) ;

			# in icosohedral (and maybe octohedral/tetraherdal) symm we have
			#	  2-fold symm operators where x==0 and must ==0 to keep other symmetries intact
			# for these cases we cannot allow x/angle_x to move
			if ( $x_dist > 1e-3 ) {
				if ($nsiblings == 2) {
					push @{ $dof_lines },	  "set_dof JUMP$child0_idstring x($x_dist) angle_x";
					#push @{ $dof_lines },	   "set_dof JUMPGROUP$is_controlling x angle_x";
				} elsif ($nsiblings > 2) {
					push @{ $dof_lines },	  "set_dof JUMP$child0_idstring x($x_dist)";
					#push @{ $dof_lines },	   "set_dof JUMPGROUP$is_controlling x";
				}
			}
		}

		foreach my $child_idx ( 1..$nchildren-1 ) {
			my $child_idstring = $tree->{CHILDREN}->[$child_idx]->{PATH};
			push @{ $connect_lines }, "connect_virtual JUMP$child_idstring VRT$child0_idstring VRT$child_idstring";
		}
	} else { # $nchildren == 0
		##
		## now vrts + jumps for child nodes
		# 1 -- jump to the COM
		# is this jump the controlling jump?
		my $is_controlling = -1;
		my $depth = 0;
		foreach my $nodes_i (@{ $nodes_by_depth }) {
			if ($id_string eq $nodes_i->[0]) {
				$is_controlling = $depth;
			}
			$depth++;
		}
		push @{ $connect_lines }, "connect_virtual JUMP".$id_string."_to_com VRT$id_string VRT$id_string"."_base";

		if ($is_controlling >= 1) {
			my $x_dist = vnorm( vsub( $parent_com,$origin) ) ;
			# in icosohedral (and maybe octohedral/tetraherdal) symm we have
			#	  2-fold symm operators where x==0 and must ==0 to keep other symmetries intact
			# for these cases we cannot allow x/angle_x to move
			if ( $x_dist > 1e-3 ) {
				if ($nsiblings == 2) {
					push @{ $dof_lines },	  "set_dof JUMP$id_string"."_to_com x($x_dist) angle_x";
				} elsif ($nsiblings > 2) {
					push @{ $dof_lines },	  "set_dof JUMP$id_string"."_to_com x($x_dist)";
				}
			}
		}

		# 2 -- if an interface res, jump to the subunit
		if (defined $connected_subunits->{ $tree->{PATH} }) {
			# if this is "subunit 0"
			if ($connected_subunits->{ $tree->{PATH} } == 0) {
				push @{ $dof_lines },	  "set_dof JUMP".$id_string."_to_subunit angle_x angle_y angle_z";
			}

			push @{ $connect_lines }, "connect_virtual JUMP".$id_string."_to_subunit VRT$id_string"."_base SUBUNIT";

		}


		push @{ $vrt_lines }, "xyz VRT$id_string"."_base  ".
					sprintf("%.7f,%.7f,%.7f", $myX->[0], $myX->[1], $myX->[2])."  ".
					sprintf("%.7f,%.7f,%.7f", $myY->[0], $myY->[1], $myY->[2])."  ".
					sprintf("%.7f,%.7f,%.7f", $origin->[0], $origin->[1], $origin->[2]);

	}
}

##########
## my $x = get_com( $subtree )
##	  get the center of mass of a subtree
##	  do this by summing the CoM's of all subunits
sub get_com {
	my $tree = shift;
	return $tree->{T};
}


###################################################################################
###################################################################################
###
### Kabsch fast RMS alignment
###
###################################################################################
###################################################################################

# my ($R,$rmsd, $Ycom, $Ycom_to_Xcom) = rms_align( $x,$y );
sub rms_align {
	my ($X,$Y) = @_;

	my ($nlist,$mov_com, $mov_to_ref, $R, $E0) = setup_rotation( $X, $Y );
	my ($U,$residual) = calculate_rotation_matrix($R,$E0);

	my $rmsd = sqrt( fabs($residual*2.0/$nlist) );
	return ($U,$rmsd,$mov_com, $mov_to_ref);
}

# my $com = recenter( $x );
sub recenter {
	my ($X) = @_;
	my $Natms = scalar(@{ $X });
	my $com = [0,0,0];

	foreach my $n (0..$Natms-1) {
		foreach my $i (0..2) {
			$com->[$i] += $X->[$n][$i];
		}
	}

	foreach my $i (0..2) {
		$com->[$i] /= $Natms;
	}

	foreach my $n (0..$Natms-1) {
		foreach my $i (0..2) {
			$X->[$n][$i] -= $com->[$i];
		}
	}
	return $com;
}


# normalize($a)
sub normalize {
	my $a = shift;
	my $b = sqrt($a->[0]*$a->[0] + $a->[1]*$a->[1] + $a->[2]*$a->[2]);
	if ($b > 1e-6) {
		$a->[0] /= $b; $a->[1] /= $b; $a->[2] /= $b;
	}
}


# my $a_dot_b = dot($a,$b)
sub dot {
	my ($a,$b) = @_;
	return ($a->[0]*$b->[0] + $a->[1]*$b->[1] + $a->[2]*$b->[2]);
}


# my $a = cross ( b , c )
sub cross {
	my ($b,$c) = @_;
	my $a = [ $b->[1]*$c->[2] - $b->[2]*$c->[1] ,
			  $b->[2]*$c->[0] - $b->[0]*$c->[2] ,
			  $b->[0]*$c->[1] - $b->[1]*$c->[0] ];
	return $a;
}


# ($nlist,$mov_com, $mov_to_ref, $R, $E0) = setup_rotation( $ref_xlist, $mov_xlist )
sub setup_rotation {
	my ( $ref_xlist, $mov_xlist ) = @_;

	my $nlist = min( scalar(@{ $ref_xlist }) , scalar(@{ $mov_xlist }) );
	my $ref_com = [0,0,0];
	my $mov_com = [0,0,0];
	my $mov_to_ref = [0,0,0];

	foreach my $n (0..$nlist-1) {
		foreach my $i (0..2) {
			$mov_com->[$i] += $mov_xlist->[$n][$i];
			$ref_com->[$i] += $ref_xlist->[$n][$i];
		}
	}
	foreach my $i (0..2) {
		$mov_com->[$i] /= $nlist;
		$ref_com->[$i] /= $nlist;
		$mov_to_ref->[$i] = $ref_com->[$i] - $mov_com->[$i];
	}


	# shift mov_xlist and ref_xlist to centre of mass */
	foreach my $n (0..$nlist-1) {
		foreach my $i (0..2) {
			$mov_xlist->[$n][$i] -= $mov_com->[$i];
			$ref_xlist->[$n][$i] -= $ref_com->[$i];
		}
	}

	# initialize
	my $R = [ [0,0,0] , [0,0,0] , [0,0,0] ];
	my $E0 = 0.0;

	foreach my $n (0..$nlist-1) {
		# E0 = 1/2 * sum(over n): y(n)*y(n) + x(n)*x(n)
		foreach my $i (0..2) {
		  $E0 +=  $mov_xlist->[$n][$i] * $mov_xlist->[$n][$i]
				+ $ref_xlist->[$n][$i] * $ref_xlist->[$n][$i];
		}

		# R[i,j] = sum(over n): y(n,i) * x(n,j)
		foreach my $i (0..2) {
			foreach my $j (0..2) {
			   $R->[$i][$j] += $mov_xlist->[$n][$i] * $ref_xlist->[$n][$j];
			}
		}
	}
	$E0 *= 0.5;

	return ($nlist,$mov_com, $mov_to_ref, $R, $E0);
}

# helper funct
sub j_rotate {
	my ($a,$i,$j,$k,$l,$s,$tau) = @_;
	my $g = $a->[$i][$j];
	my $h = $a->[$k][$l];
	$a->[$i][$j] = $g-$s*($h+$g*$tau);
	$a->[$k][$l] = $h+$s*($g-$h*$tau);
}

# ($d,$v,$nrot) = jacobi3($a)
#	 computes eigenval and eigen_vec of a real 3x3
#	 symmetric matrix. On output, elements of a that are above
#	 the diagonal are destroyed. d[1..3] returns the
#	 eigenval of a. v[1..3][1..3] is a matrix whose
#	 columns contain, on output, the normalized eigen_vec of a.
#	 n_rot returns the number of Jacobi rotations that were required
sub jacobi3 {
	my $a = shift;

	my $v = [ [1,0,0] , [0,1,0] , [0,0,1] ];
	my $b = [ $a->[0][0] , $a->[1][1] , $a->[2][2] ];
	my $d = [ $a->[0][0] , $a->[1][1] , $a->[2][2] ];
	my $z = [0,0,0];
	my $n_rot = 0;
	my $thresh = 0;

	# 50 tries!
	foreach my $count (0..49) {

		# sum off-diagonal elements
		my $sum = fabs($a->[0][1])+fabs($a->[0][2])+fabs($a->[1][2]);

		# if converged to machine underflow
		if ($sum == 0.0) {
		  return($d,$v,$n_rot);
		}

		# on 1st three sweeps..
		my $thresh = 0;
		if ($count < 3) {
			$thresh = $sum * 0.2 / 9.0;
		}

		foreach my $i (0,1) {
			foreach my $j ($i+1..2) {
				my $g = 100.0 * fabs($a->[$i][$j]);

				# after four sweeps, skip the rotation if
				# the off-diagonal element is small
				if ( $count > 3
					  && fabs($d->[$i])+$g == fabs($d->[$i])
					  && fabs($d->[$j])+$g == fabs($d->[$j]) ) {
					$a->[$i][$j] = 0.0;
				} elsif (fabs($a->[$i][$j]) > $thresh) {
					my $h = $d->[$j] - $d->[$i];
					my ($t,$s,$tau,$theta);

					if (fabs($h)+$g == fabs($h)) {
						$t = $a->[$i][$j] / $h;
					} else {
						$theta = 0.5 * $h / ($a->[$i][$j]);
						$t = 1.0 / ( fabs($theta) + sqrt(1.0 + $theta*$theta) );
						if ($theta < 0.0) { $t = -$t; }
					}

					my $c = 1.0 / sqrt(1 + $t*$t);
					$s = $t * $c;
					$tau = $s / (1.0 + $c);
					$h = $t * $a->[$i][$j];

					$z->[$i] -= $h;
					$z->[$j] += $h;
					$d->[$i] -= $h;
					$d->[$j] += $h;

					$a->[$i][$j] = 0.0;

					foreach my $k (0..$i-1) {
						j_rotate($a, $k, $i, $k, $j, $s, $tau);
					}
					foreach my $k ($i+1..$j-1) {
						j_rotate($a, $i, $k, $k, $j, $s, $tau);
					}
					foreach my $k ($j+1..2) {
						j_rotate($a, $i, $k, $j, $k, $s, $tau);
					}
					foreach my $k (0..2) {
						j_rotate($v, $k, $i, $k, $j, $s, $tau);
					}
					$n_rot++;
				}
			}
		}

		foreach my $i (0..2) {
			$b->[$i] += $z->[$i];
			$d->[$i] = $b->[$i];
			$z->[$i] = 0.0;
		}
	}

	print STDERR "WARNING: Too many iterations in jacobi3!	You're bad and you should feel bad.\n";
	exit -1;
}


# ($eigen_vec, $eigenval) = diagonalize_symmetric( $matrix )
sub diagonalize_symmetric {
	my $matrix = shift;
	my $n_rot = 0;

	my ($eigenval,$vec,$n_rot) = jacobi3($matrix);

	# sort solutions by eigenval
	foreach my $i (0..2) {
		my $k = $i;
		my $val = $eigenval->[$i];

		foreach my $j ($i+1..2) {
			if ($eigenval->[$j] >= $val) {
				$k = $j;
				$val = $eigenval->[$k];
			}
		}

		if ($k != $i) {
			$eigenval->[$k] = $eigenval->[$i];
			$eigenval->[$i] = $val;
			foreach my $j (0..2) {
				$val = $vec->[$j][$i];
				$vec->[$j][$i] = $vec->[$j][$k];
				$vec->[$j][$k] = $val;
			}
		}
	}

	# transpose
	my $eigen_vec = [ [$vec->[0][0],$vec->[1][0],$vec->[2][0]] ,
					  [$vec->[0][1],$vec->[1][1],$vec->[2][1]] ,
					  [$vec->[0][2],$vec->[1][2],$vec->[2][2]] ];
	return ($eigen_vec, $eigenval);
}


# ($U,$residual) = calculate_rotation_matrix($R,$E0)
sub calculate_rotation_matrix {
	my ($R,$E0) = @_;

	my $RtR = [ [0,0,0] , [0,0,0] , [0,0,0] ];
	my $left_eigenvec = [ [0,0,0] , [0,0,0] , [0,0,0] ];
	my $right_eigenvec = [ [0,0,0] , [0,0,0] , [0,0,0] ];
	my $eigenval = 0;

	 # Rt <- transpose of R
	my $Rt = [ [$R->[0][0],$R->[1][0],$R->[2][0]] ,
			   [$R->[0][1],$R->[1][1],$R->[2][1]] ,
			   [$R->[0][2],$R->[1][2],$R->[2][2]] ];

	# make symmetric RtR = Rt X R
	foreach my $i (0..2) {
		foreach my $j (0..2) {
			$RtR->[$i][$j] = 0.0;
			foreach my $k (0..2) {
				$RtR->[$i][$j] += $Rt->[$k][$i] * $R->[$j][$k];
			}
		}
	}

	($right_eigenvec, $eigenval) = diagonalize_symmetric( $RtR );

	# right_eigenvec's should be an orthogonal system but could be left
	#	or right-handed. Let's force into right-handed system.
	$right_eigenvec->[2] = cross($right_eigenvec->[0], $right_eigenvec->[1]);

	# From the Kabsch algorithm, the eigenvec's of RtR
	#	are identical to the right_eigenvec's of R.
	#	This means that left_eigenvec = R x right_eigenvec
	foreach my $i (0..2) {
		foreach my $j (0..2) {
			$left_eigenvec->[$i][$j] = dot($right_eigenvec->[$i], $Rt->[$j]);
		}
	}

	foreach my $i (0..2) {
		normalize($left_eigenvec->[$i]);
	}

	# Force left_eigenvec[2] to be orthogonal to the other vectors.
	# First check if the rotational matrices generated from the
	#	orthogonal eigenvectors are in a right-handed or left-handed
	#	coordinate system - given by sigma. Sigma is needed to
	#	resolve this ambiguity in calculating the RMSD.
	my $sigma = 1.0;
	my $v = cross($left_eigenvec->[0], $left_eigenvec->[1]);
	if (dot($v, $left_eigenvec->[2]) < 0.0) {
		$sigma = -1.0;
	}
	foreach my $i (0..2) {
		$left_eigenvec->[2][$i] = $v->[$i];
	}

	# calc optimal rotation matrix U that minimises residual
	my $U = [ [0,0,0] , [0,0,0] , [0,0,0] ];
	foreach my $i (0..2) {
		foreach my $j (0..2) {
			foreach my $k (0..2) {
				$U->[$i][$j] += $left_eigenvec->[$k][$i] * $right_eigenvec->[$k][$j];
			}
		}
	}

	my $residual = $E0 - sqrt(fabs($eigenval->[0]))
					   - sqrt(fabs($eigenval->[1]))
					   - $sigma * sqrt(fabs($eigenval->[2]));

	return ($U,$residual);
}


###################################################################################
###################################################################################
###
### Vector and matrix ops
###
###################################################################################
###################################################################################

sub deep_copy {
	my $this = shift;
	if (not ref $this) {
		$this;
	} elsif (ref $this eq "ARRAY") {
		[map deep_copy($_), @$this];
	} elsif (ref $this eq "HASH") {
		+{map { $_ => deep_copy($this->{$_}) } keys %$this};
	} else { die "what type is $_?" }
}


# rotation from euler angles
sub euler {
	my ($aa, $bb, $gg) = @_;
	my $MM;

	$MM->[0][0] = (-sin($aa)*cos($bb)*sin($gg) + cos($aa)*cos($gg));
	$MM->[0][1] = ( cos($aa)*cos($bb)*sin($gg) + sin($aa)*cos($gg));
	$MM->[0][2] = ( sin($bb)*sin($gg));
	$MM->[1][0] = (-sin($aa)*cos($bb)*cos($gg) - cos($aa)*sin($gg));
	$MM->[1][1] = ( cos($aa)*cos($bb)*cos($gg) - sin($aa)*sin($gg));
	$MM->[1][2] = ( sin($bb)*cos($gg));
	$MM->[2][0] = ( sin($aa)*sin($bb));
	$MM->[2][1] = (-cos($aa)*sin($bb));
	$MM->[2][2] = ( cos($bb));

	return $MM;
}

# my ($X,$Y,$Z,$W)=R2quat($M)
sub R2quat {
	my $R = shift;
	my ($S,$X,$Y,$Z,$W);
	if ( $R->[0][0] > $R->[1][1] && $R->[0][0] > $R->[2][2] )  {
		$S	= sqrt( 1.0 + $R->[0][0] - $R->[1][1] - $R->[2][2] ) * 2;
		$X = 0.25 * $S;
		$Y = ($R->[1][0] + $R->[0][1] ) / $S;
		$Z = ($R->[2][0] + $R->[0][2] ) / $S;
		$W = ($R->[2][1] - $R->[1][2] ) / $S;
	} elsif ( $R->[1][1] > $R->[2][2] ) {
		$S	= sqrt( 1.0 + $R->[1][1] - $R->[0][0] - $R->[2][2] ) * 2;
		$X = ($R->[1][0] + $R->[0][1] ) / $S;
		$Y = 0.25 * $S;
		$Z = ($R->[2][1] + $R->[1][2] ) / $S;
		$W = ($R->[0][2] - $R->[2][0] ) / $S;
	} else {
		$S	= sqrt( 1.0 + $R->[2][2] - $R->[0][0] - $R->[1][1] ) * 2;
		$X = ($R->[0][2] + $R->[2][0] ) / $S;
		$Y = ($R->[2][1] + $R->[1][2] ) / $S;
		$Z = 0.25 * $S;
		$W = ($R->[1][0] - $R->[0][1]) / $S;
	}
	return ($X,$Y,$Z,$W);
}

# my ($R)=R2quat($X,$Y,$Z,$W)
sub quat2R {
	my ($X,$Y,$Z,$W) = @_;
	my $xx = $X * $X; my $xy = $X * $Y; my $xz = $X * $Z;
	my $xw = $X * $W; my $yy = $Y * $Y; my $yz = $Y * $Z;
	my $yw = $Y * $W; my $zz = $Z * $Z; my $zw = $Z * $W;
	my $R = [ [ 1 - 2 * ( $yy+$zz ) ,	  2 * ( $xy-$zw ) ,		2 * ( $xz+$yw ) ] ,
			  [		2 * ( $xy+$zw ) , 1 - 2 * ( $xx+$zz ) ,		2 * ( $yz-$xw ) ] ,
			  [		2 * ( $xz-$yw ) ,	  2 * ( $yz+$xw ) , 1 - 2 * ( $xx+$yy ) ] ];
	return $R;
}

# my ($R)=R2quat($X,$Y,$Z,$W)
sub quatnorm {
	my ($X,$Y,$Z,$W) = @_;
	my $S = sqrt( $X*$X+$Y*$Y+$Z*$Z+$W*$W );
	return [ $X/$S , $Y/$S , $Z/$S , $W/$S ];
}

#####################################
#####################################

# vector addition
sub vstr {
	my ($x) = @_;
	return "[".($x->[0]).",".($x->[1]).",".($x->[2])."]";
}

# vector addition
sub vadd {
	my ($x, $y) = @_;
	return [ $x->[0]+$y->[0], $x->[1]+$y->[1], $x->[2]+$y->[2] ];
}

# vector subtraction
sub vsub {
	my ($x, $y) = @_;
	return [ $x->[0]-$y->[0], $x->[1]-$y->[1], $x->[2]-$y->[2] ];
}

# mult vector by scalar
sub vscale {
	my ($x, $y) = @_;
	return [ $x*$y->[0], $x*$y->[1], $x*$y->[2] ];
}

# "min mod"
sub minmod {
	my ($x,$y) = @_;
	my $r = fmod($x,$y);
	if ($r < -fabs( $y/2.0 ) ) { $r += fabs( $y ); }
	elsif ($r >	 fabs( $y/2.0 ) ) { $r -= fabs( $y ); }
	return $r;
}

# vector min-modulus
sub vminmod {
	my ($x,$y) = @_;
	return [ minmod($x->[0],$y->[0]), minmod($x->[1],$y->[1]), minmod($x->[2],$y->[2]) ];
}


#####################################
#####################################

# raise a matrix to a power
# dumb way of doing it
sub mpow {
	my ($mat, $pow) = @_;
	my $matpow = $mat;
	foreach my $i (2..$pow) {
		$matpow = mmult( $mat, $matpow );
	}
	return $matpow;
}

# matrix x vector mult
sub mapply {
	my ($rotmat, $cart) = @_;
	my $out = [0, 0, 0];
	my ($i, $j);
	for ($i=0; $i < 3; ++$i) {
		for ($j=0; $j < 3; ++$j) {
			$out->[$i] += $rotmat->[$i][$j] * $cart->[$j];
		}
	}
	return $out;
}

# matrix x matrix mult
sub mmult {
	my ($m1, $m2) = @_;
	my $out = [ [0,0,0], [0,0,0], [0,0,0] ];
	my ($i, $j, $k);
	for ($i=0; $i<3; ++$i) {
		for ($j=0; $j<3; ++$j) {
			for ($k=0; $k<3; ++$k) {
				$out->[$i][$j] += $m1->[$i][$k] * $m2->[$k][$j];
			}
		}
	}
	return $out;
}


# matrix inversion
sub minv {
	my $M = shift;
	my $Minv = [ [1,0,0] , [0,1,0] , [0,0,1] ];
	my $D = $M->[0][0] * ( $M->[1][1]*$M->[2][2] - $M->[2][1]*$M->[1][2] ) -
			$M->[0][1] * ( $M->[1][0]*$M->[2][2] - $M->[1][2]*$M->[2][0] ) +
			$M->[0][2] * ( $M->[1][0]*$M->[2][1] - $M->[1][1]*$M->[2][0] );
	if ($D == 0)  {
		print STDERR "ERROR ... Inversion of singular matrix!\n";
		exit -1;
	}

	$Minv->[0][0] =	 ($M->[1][1]*$M->[2][2]-$M->[1][2]*$M->[2][1])/$D;
	$Minv->[0][1] = -($M->[0][1]*$M->[2][2]-$M->[0][2]*$M->[2][1])/$D;
	$Minv->[0][2] =	 ($M->[0][1]*$M->[1][2]-$M->[0][2]*$M->[1][1])/$D;
	$Minv->[1][0] = -($M->[1][0]*$M->[2][2]-$M->[2][0]*$M->[1][2])/$D;
	$Minv->[1][1] =	 ($M->[0][0]*$M->[2][2]-$M->[0][2]*$M->[2][0])/$D;
	$Minv->[1][2] = -($M->[0][0]*$M->[1][2]-$M->[0][2]*$M->[1][0])/$D;
	$Minv->[2][0] =	 ($M->[1][0]*$M->[2][1]-$M->[2][0]*$M->[1][1])/$D;
	$Minv->[2][1] = -($M->[0][0]*$M->[2][1]-$M->[0][1]*$M->[2][0])/$D;
	$Minv->[2][2] =	 ($M->[0][0]*$M->[1][1]-$M->[0][1]*$M->[1][0])/$D;

	return $Minv;
}

# matrix determinant
sub mdet {
	my $M = shift;
	my $D = $M->[0][0] * ( $M->[1][1]*$M->[2][2] - $M->[2][1]*$M->[1][2] ) -
			$M->[0][1] * ( $M->[1][0]*$M->[2][2] - $M->[1][2]*$M->[2][0] ) +
			$M->[0][2] * ( $M->[1][0]*$M->[2][1] - $M->[1][1]*$M->[2][0] );
	return $D;
}

# vector norm
sub vnorm {
	my $x = shift;
	return sqrt( ($x->[0]*$x->[0]) + ($x->[1]*$x->[1]) + ($x->[2]*$x->[2]) );
}

# vector norm^2
sub vnorm2 {
	my $x = shift;
	return ( ($x->[0]*$x->[0]) + ($x->[1]*$x->[1]) + ($x->[2]*$x->[2]) );
}

# cart distance
sub vdist {
	my ($x1, $x2) = @_;
	return sqrt (
			   ($x1->[0]-$x2->[0])*($x1->[0]-$x2->[0]) +
			   ($x1->[1]-$x2->[1])*($x1->[1]-$x2->[1]) +
			   ($x1->[2]-$x2->[2])*($x1->[2]-$x2->[2])
				);
}

# are two transformations the inverso of one another
sub is_inverse {
	my $tol = 1e-8;
	my ( $R_i,$T_i, $R_j,$T_j ) = @_;

	my $testR = mmult( $R_i , $R_j );
	my $testT = vadd( mapply( $R_j,$T_i ) , $T_j );

	my $errR = square($testR->[0][0]-1) + square($testR->[1][1]-1) + square($testR->[2][2]-1) +
			   square($testR->[0][1])	+ square($testR->[0][2])   + square($testR->[1][2]) +
			   square($testR->[1][0])	+ square($testR->[2][0])   + square($testR->[2][1]);
	my $errT = square($testT->[0])+square($testT->[1])+square($testT->[2]);

#print " (0) $errR	 $errT\n";
	if ($errR < $tol && $errT < $tol) { return 1; }
	return 0;
}

#
sub is_identity {
	my $testR = shift;
	my $tol = 1e-8;
	if (scalar( @_ ) >= 1) {
		$tol = shift;
	}
	my $errR = square($testR->[0][0]-1) + square($testR->[1][1]-1) + square($testR->[2][2]-1) +
			   square($testR->[0][1])	+ square($testR->[0][2])   + square($testR->[1][2]) +
			   square($testR->[1][0])	+ square($testR->[2][0])   + square($testR->[2][1]);

	if ($errR < $tol) { return 1; }
	return 0;
}

# is the transform (Rn,Tn) equivalent to the transform (Ri,Ti)->(Rj,Tj)
sub is_equivalent {
	my $tol = 1e-8;
	my ( $R_n,$T_n, $R_i,$T_i, $R_j,$T_j ) = @_;

	my $R_i_inv = minv( $R_i );
	my $T_i_inv = [ -$T_i->[0], -$T_i->[1], -$T_i->[2] ];

	my $R_test = mmult( $R_i_inv, $R_j );
	my $T_test = mapply( $R_i_inv, vsub( $T_j, $T_i ) );

	my $errR = square($R_test->[0][0]-$R_n->[0][0])+square($R_test->[0][1]-$R_n->[0][1])+square($R_test->[0][2]-$R_n->[0][2])
			 + square($R_test->[1][0]-$R_n->[1][0])+square($R_test->[1][1]-$R_n->[1][1])+square($R_test->[1][2]-$R_n->[1][2])
			 + square($R_test->[2][0]-$R_n->[2][0])+square($R_test->[2][1]-$R_n->[2][1])+square($R_test->[2][2]-$R_n->[2][2]);
	my $errT = square($T_test->[0]-$T_n->[0])+square($T_test->[1]-$T_n->[1])+square($T_test->[2]-$T_n->[2]);

	##
	## don't think we need to check T ...
	if ($errR < $tol) { return 1; }
	#if ($errR < $tol && $errT < $tol) { return 1; }
	return 0;
}

###########
# f2c/c2f #
###########

sub d2r { return (@_[0]*PI/180); }
sub square { return @_[0]*@_[0]; }

# my ($f2c,$c2f) = crystparams($a,$b,$c,$alpha,$beta,$gamma)
sub crystparams {
	my ($a,$b,$c,$alpha,$beta,$gamma) = @_;

	if ($a*$b*$c == 0) {
		print STDERR "Must provide valid crystal parameters!\n";
		exit -1;
	}

	my $f2c = [ [0,0,0] , [0,0,0] , [0,0,0] ];

	my $ca = cos(d2r($alpha)); my $cb = cos(d2r($beta)); my $cg = cos(d2r($gamma));
	my $sa = sin(d2r($alpha)); my $sb = sin(d2r($beta)); my $sg = sin(d2r($gamma));
	$f2c->[0][0] = $a;
	$f2c->[0][1] = $b * $cg;
	$f2c->[0][2] = $c * $cb;
	$f2c->[1][1] = $b * $sg;
	$f2c->[1][2] = $c * ($ca - $cb*$cg) / $sg;
	$f2c->[2][2] = $c * $sb * sqrt(1.0 - square(($cb*$cg - $ca)/($sb*$sg)));

	my $c2f = minv($f2c);

	return ($f2c,$c2f);
}


#######
# end #
#######
