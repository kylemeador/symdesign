// ERRAT_V2 - v1.0 - last modified on 9/5/2002
//
// Copyright 2002.  Dennis Obukhov and Todd Yeates,
// UCLA Department of Chemistry and Biochemistry,
// Los Angeles, CA.
//
// This program may be distributed freely as long as
// the header remains intact.  This program may not
// be incorporated into other software packages
// without the consent of the authors.
//
// Please cite:
// Colovos, C. and Yeates, T.O.  1993. Verification of
// Protein Structures: Patterns of Non-Bonded Contacts.
// Prot. Sci. 2, 1511-1519.

// Kyle Meador KM
// This version was modified for the sole purpose of
// returning ERRAT total and per residue scores as
// SymDesign metric queries


#include <iostream>
#include <fstream>
#include <string>
#include <cstring>
#include <stdlib.h>
#include <math.h>
#include <ctype.h>
#include <iomanip>

using std::cout;
using std::endl;
using std::ifstream;
using std::ofstream;

//KM patch for large memory allocation
#include <algorithm>
#include <vector>
#include <stdio.h>
#include <sys/poll.h>

using std::vector;
//check variables size/bxmx in arrays
// check filename


double matrixdb (double matrix[6]);
// add prototype
// add double matrix[5];
// add call at the end
// add f(x) - REMOVE ALL STREAMS AND MULTIFILE
// temp[1] -> matrix[1]
// remove and streams mout, char chain[i] and chain determination;
// change zout outp
#define TIMEOUT 1000 // 1 sec

int main(int argc, char* argv[]){//1
    int flag2=0;//test for too many atoms in a box
    int flag3=0;
    int	kadd, o=0, gg=0;
    // COMMAND LINE ARGS STUFF HERE
	char file[1000], logfilename[1000], psfilename[1000], seed[1000], path[1000];
	//char tyfile[100];

    //STREAMS
	ifstream count_in;// input filenames Directory
	ifstream fin;// input files PDB
	ofstream fout;// output logs to files
	ofstream zout;//outputs results for analysis
	ofstream mout;//outputs file progress report
	//ofstream tyout; // T.O.Y.
	//ofstream err;

	//STDIN reading
	vector<std::string> all_lines;
	struct pollfd fds;
    int ret;
    fds.fd = 0; /* this is STDIN */
    fds.events = POLLIN;
    ret = poll(&fds, 1, TIMEOUT);
	if(argc == 2 && ret == 1) {
        strcpy(path, argv[1]);
        strcpy(logfilename, path);
        //strcpy(psfilename, path);
        strcat(logfilename, "errat.log");
        //strcat(psfilename, "errat.ps");
        //if(ret == 1)
        //    ;//Yes stdin
        //else if(ret == 0)
        //    ;//No stdin
        //else
        //    ;// Error
        //return 0;
        //		{
        for (std::string line; std::getline(std::cin, line);) {
            all_lines.push_back(line);
        //std::cout << line << std::endl;
        }//KM
        strcpy(file, "stdin");
    }
    else if(argc != 3) {
        if (argc ==2 && ret != 1){
            puts("\nFailed to grab stdin from shell! Ensure you piped the pdb file to ./errat localpath\n");
            exit(1);
        }
        else{puts("\n2 arguments required: ./errat pdbid localpath\n"); exit(1); }
    }
    else {
        strcpy(seed, argv[1]);
	    strcpy(path, argv[2]);
        // file is the pdb file name, logf is the output logfile
        strcpy(file, path);
        strcpy(logfilename, path);
        //strcpy(psfilename, path);
        //strcpy(tyfile, path);
        strcat(file, seed);
        strcat(logfilename, seed);
        //strcat(psfilename, seed);
        //strcat(tyfile, seed);
        strcat(file, ".pdb");
        strcat(logfilename, ".logf");
        //strcat(psfilename, ".ps");
        //strcat(tyfile, ".ty");
        fin.open(file) ;	//read from that input file
		if (fin.fail()!=0){
			cout << "Failed opening file " << file << endl;
			gg=1;
			exit(1);}
		else{
		    for (char char_line[100]; fin.getline(char_line, 100);) {
		        std::string line(char_line);
                all_lines.push_back(line);
                //cout << line << std::endl;
            }//KM
        }
		zout << o <<"	"<< file << endl<<endl;
    }
    //cout<<" FILES "<<file<< " = "<<logfilename<< endl;//" = "<<psfilename<<"\n\n";

// PROBLEM WITH IOS
//	err.setf(ios::fixed);
//	err.setf(ios::showpoint);
//

	//err << std::fixed;
	//err << std::showpoint;
	//err << std::setprecision(3);

		//command doesn't operate under VC++.
/*	count_in.open("pdb.txt") ;	//read from Directory
	if (count_in.fail())
	{	cout << "Failed opening pdb.txt"<< file;
		exit (1);}
*/	//NOTE: PDBLIST.TXT MUST END ON LAST PDB W/O HARD RETURN, ELSE DOUBLE COUNTING OF THE LAST FILE!

	fout.open(logfilename);//write to a log file
	if (fout.fail()){
		cout << "Failed opening log.txt" << file;
		exit (1);
	}

	/*zout.open("framesf.txt");//writes results
	if (zout.fail())
	{	cout << "Failed opening frames.txt"<< file;
		exit (1);}*/

	/*mout.open("fstat.txt") ;	//write to a log file
	if (fout.fail())
	{	cout << "Failed opening fstat.txt"<< file;
		exit (1);}*/

	//err.open(psfilename) ;	//write to a postscript output file
	//if (err.fail())
	//{	cout << "Failed opening err.txt"<< file;
	//	exit (1);}

	//tyout.open(tyfile) ;	//write to a log file
	//if (fout.fail())
	//{	cout << "Failed opening .ty"<< file;
	//	exit (1);}

	double lmt[3];
	/*ifstream sts;
	sts.open("lmt.txt");
	if (sts.fail())
	{	cout << "Failed opening lmt.txt"<< endl;
		exit (1);}

	sts >> lmt[2];
	sts >> lmt[1];
	fout <<"Limits:"<<lmt[1]<<"	"<<lmt[2]<< endl;
	sts.close();*/
	lmt[2]=11.526684477428809;
	lmt[1]=17.190823041860433;

/*	cout << "Main Menu:"<<endl;
	cout << "Enter (1) to analyse the PDBs from the local directory."<<endl;
	cout << "Enter (2) to analyse the PDBs from the database."<<endl;
	cin >> fl;
	fl=1;
*/	//if (fl==1)
		{
		//	cin >> fl2;
		}
		//if ( (fl!=1)||(fl!=2) )
		{
			//exit(1);
		}

	//Begins Processing
//	cout << "PROCESSING:"<< endl;

//	atmnum = 0;//new file
//	resnum[0]=0;//new file
//	lowframe = 0; //new file

//	while (! count_in.eof())//FILENAME LOOP
	{//2

		flag2=0;// skip this pdb analysis of many atoms/box or boxes or atoms;
		flag3=0;

/*		count_in.getline(flnm, 100);	//input filename from directory

		if (fl==2)
		{			for (j=0; j<10; j++) {file[j+8]=tolower(flnm[j]);}
					//chain=flnm[4];
					//fout << "chain "<<chain<<endl;
					//if (flnm[4]=='0')
					//{chain=' ';}
					//ff = strlen(file);
					file[12]='.';
					file[13]='e';
					file[14]='n';
					file[15]='t';
					file[16]='\0';// takes 9 characters
		}
		else if (fl==1)
		{
			for (j=0; j<12; j++) {file[j]=tolower(flnm[j]);}
			file[13]='\0';
		}
		//if (fl==2)
		{
			o=o++;//counter of *.PDBs processed
			cout << o <<"	"<< file << "\n";
			fout << o <<"	"<< file << endl;
		}
*/
		kadd = 0;// new file init
		//gg=0;// new filename test

		/*if (fl==1)
		{
			fin.open(fl2) ;	//read from that input file
			if (fin.fail()!=0)
			{	cout << "Failed opening file " << fl2 << endl;
				gg=1;}
			zout <<"	"<< fl2 << endl<<endl;
		}*/
		//if (fl==2)
        const int size = all_lines.size();//KM
        //KM move initialization of all variables below here to dynamically utilize the array size
        //all arrays upon their import to UNIX must be modified to ensure proper load.
        //STAGE 1 VARS
        //char	file [100] = "pdb.pdb\0";
        //char	flnm [100] = "pdb.pdb";
        char	line [100];
        int	fl;
        //char fl2[100];
        char chain;
        const char	atom_test [7] = {'A','T','O','M',' ',' ','\0'};
        int atom;
        char	line2 [7];
        //const int size=250000;
        //const int bxmx=200000;
        int		i,j,k,l,m,n,p,q,r,s,v, aa, ab;//, o=0, gg=0; //for loops
        int		atmnum;
        char	name_temp; char name_temp2[4];
        //int	name[size]; int bnam[size];
        vector<int>	name; int bnam[size];
        name.push_back(0); //Todd uses all 1 indexed arrays...
        char	altLoc;
        char	resName[4];
        char	chainID[size];
        char	resSeq_temp[5];
        int	resSeq[size];
        int	resnum[size];
        char	x[9], y[9], z[9];
        double 	xyz[3][size];
        int		flag = 0;
        int		kadd;
        const int chaindif=10000;

        //STAGE 2 VARS
        double min[4]={0,0,0,0}, max[4]={0,0,0,0};

        //STAGE 3 VARS

        int nbx[4];
        const double boxsize=4;// must be double for calculations
        //int ibox1[16][bxmx];
        int temp;
        int ix, iy, iz;
        int ind;// must give ceil, so int
        int most=0;
        //int flag2=0;//test for too many atoms in a box
        //int flag3=0;

        //STAGE 4 VARS

        int rer;
        const double	radius=3.75, rsq=radius*radius;//up front? useful in this step
        const double	radmin=3.25, ssq=radmin*radmin;
        const int		ndelta= ceil(radius/boxsize); // 1
        double dsq;		//holds squared distance

        int jbx, jby, jbz;// target location
        int ibz1,ibz2,iby1,iby2,ibx1,ibx2;//min/max boxes

        double temp1;//store the sqrt of the distance
        double temp2;//insert name[n] into array

        int count=0;//counts interactions
        int lowframe=0;
        const double maxwin=100.694; //cutoff of interactions in the frame
        double c[4][4];
        double matrix[6]; //input of data into function
        double mtrx, mtrxstat;
        double mstat;
        double pstat, stat;//tells us of the fraction of frames that is above 95%.

        //POSTSCRIPT
        // double errat[size];
        vector<double> errat;
        errat.push_back(0);//KM Because of the indexing in the post script, add 0 array record
        //errat.push_back(0);//KM
        //errat.push_back(0);//KM
        //errat.push_back(0);//KM
        //errat.push_back(0);//KM Add 4 entries as the window score is 9 long and centered at residue number + 5
        int chainx;
        int ich;
        int ir1[100];
        int ir2[100];
        char id_by_chain[100];
        double ms, mst;
        double sz;
        int np, ip, ir0, ir;
        int z1, z2;
        char bar[5];

	    atmnum = 0;//new file
	    resnum[0]=0;//new file
	    lowframe = 0; //new file

		if (gg==0){//2_5
		//for (i=0;( (!(fin.eof()))&&(flag3==0)&&(flag2==0) );)//PDB LOOP
        i=0;
        char * line = new char [all_lines[i].length()+1];
		for (int line_num=0;( (line_num < all_lines.size())&&(flag3==0)&&(flag2==0) ); line_num++){//3 //PDB LOOP
            strcpy(line, all_lines[line_num].c_str());
			//fin.getline (line, 100);
			for (j=0; j<6; j++) {line2[j]=line[j];}	line2[6]='\0';
			if (strcmp (atom_test, line2) == 0){//4 //single line process
			    atom++;
				//fout << line << endl;// test

//				i++;//iteration only if get to here
//				if (i > (size-1))
//				{
//					flag3=1;
//					fout <<"ERROR: PDB WITH TOO MANY ATOMS. CUT OFF FURTHER INPUT. "<< endl;
//					//break;
//				}
//				else
//				{//5
				altLoc = line[16];
				//fout << "altLoc	17" << altLoc << endl;//test
				if (!((altLoc == ' ') || (altLoc == 'A') ||
					(altLoc == 'a') || (altLoc == 'P'))){
					fout << "Reject 2' Conformation atom#	"<<atom<< "	chain	"<< line[21] << endl;
//					i--;
//					flag=1;
                    continue;
				}

				for (j=17; j<20; j++) {resName[j-17]=line[j];}
                resName[3]='\0';
				//cout << i << "	" << resName << endl;
				if (!((strcmp (resName, "GLY\0") == 0) ||
					  (strcmp (resName, "ALA\0") == 0) ||
					  (strcmp (resName, "VAL\0") == 0) ||
					  (strcmp (resName, "LEU\0") == 0) ||
					  (strcmp (resName, "ILE\0") == 0) ||
					  (strcmp (resName, "TYR\0") == 0) ||
					  (strcmp (resName, "CYS\0") == 0) ||
					  (strcmp (resName, "MET\0") == 0) ||
					  (strcmp (resName, "TRP\0") == 0) ||
					  (strcmp (resName, "PHE\0") == 0) ||
					  (strcmp (resName, "HIS\0") == 0) ||
					  (strcmp (resName, "PRO\0") == 0) ||
					  (strcmp (resName, "SER\0") == 0) ||
					  (strcmp (resName, "THR\0") == 0) ||
					  (strcmp (resName, "LYS\0") == 0) ||
					  (strcmp (resName, "ARG\0") == 0) ||
					  (strcmp (resName, "GLU\0") == 0) ||
					  (strcmp (resName, "ASP\0") == 0) ||
					  (strcmp (resName, "GLN\0") == 0) ||
					  (strcmp (resName, "ASN\0") == 0))){
					fout <<"***Warning: Reject Nonstandard Residue - "<<resName<< endl;
					//i--;
					//flag=1;
                    continue;
				}

				name_temp = line[13];//tested
                //if		(name_temp =='C') name[i]=1;
				if		(name_temp =='C') name.push_back(1);
				//else if (name_temp =='N') name[i]=2;
				else if (name_temp =='N') name.push_back(2);
				//else if (name_temp =='O') name[i]=3;
				else if (name_temp =='O') name.push_back(3);
				//else if (name_temp =='S') name[i]=3;//!!!!!!
				//else name[i]=0;//KM removed as these atoms don't contribute to the count
				else continue;

				i++;//iteration only if get to here
				//cout << i << endl;
				//fout << "name["<<i<<"] 14	"<< name[i] << "name " << name_temp <<endl;//test

				name_temp2[0] = line[13];
				name_temp2[1] = line[14];
				name_temp2[2] = line[15];
				name_temp2[3] = '\0';
				//fout << "name_temp2 141516	" << name_temp2 << endl;//test
				if (	(strcmp(name_temp2,"N  \0")==0)||
						(strcmp(name_temp2,"C  \0")==0)	) // used for peptide bond checks
				{	bnam[i]=1;}
				else
				{	bnam[i]=0;}
				//fout << "bnam[i]	" << bnam[i] <<endl; // test

				chainID[i]=line[21];
				//fout << "chainID[i]22	" << chainID[i] << endl;//test

				for (j=22; j<26; j++) {resSeq_temp[j-22]=line[j];} resSeq_temp[4]='\0';
				//fout << "resSeq_temp	" << resSeq_temp << endl;//test
				resSeq[i]= atof(resSeq_temp);
				//fout << "resSeq["<<i<<"]2326	" << resSeq[i] << endl;//test

				for (j=30; j<38; j++) {x[j-30]=line[j];}	x[8]='\0';
				xyz [0][i] = atof(x);
				//fout <<"x[i]3138	"<<xyz[0][i]<< endl;//test

				for (j=38; j<46; j++) {y[j-38]=line[j];}	y[8]='\0';
				xyz [1][i] = atof(y);
				//fout <<"y[i]3946	"<<xyz[1][i]<< endl;//test

				for (j=46; j<54; j++) {z[j-46]=line[j];}	z[8]='\0';
				xyz [2][i] = atof(z);
				//fout <<"z[i]4754	"<<xyz[2][i]<< endl;//test

				if ((chainID[i] != chainID[i - 1]) && (i >= 2) && (flag != 1)){
					kadd++;
					fout << "INCREMENTING CHAIN (kadd) " << kadd << endl;
				}
				if ((flag != 1)){
					resnum[i] = (resSeq[i] + (kadd * chaindif));
					atmnum = i;//max lines

					//fout << "resnum[i]	"<<resnum[i]<<endl;
					//fout << "kadd	" << kadd << endl;
					//fout << "ATMNUM:	"<<atmnum <<endl;
				}
				if ((resnum[i] < resnum[i - 1]) && (chainID[i] == chainID[i - 1]) && (flag == 0) && (i >= 2)){
					fout <<"ERROR: RESNUM DECREASE. TERMINATE ANALYSIS " << resnum[i] <<" < "<< resnum[i-1] << endl;
					for (k = resnum[i - 1]; k == resnum[i]; k++){
						fout << i << endl;
						flag2=1;
						//break;
					}
				}
                if ((resnum[i] > resnum[i - 1]) && (flag == 0)){
                    errat.push_back(0);//add a blank measurement to vector
                }
				if ((i>2) && (resnum[i]!=resnum[i-1]) &&
					(chainID[i]==chainID[i-1]) && (flag==0) && ((resnum[i]-resnum[i-1])>1)){
					fout <<"WARNING: Missing Residues " << resnum[i-1] <<">>>"<< resnum[i] << endl;
                    //for (int missing_residue=1; missing_residue < resnum[i] - resnum[i - 1]; missing_residue++){
                    //    errat.push_back(0);//add a blank measurement to vector
                    //}
				}
				// errat[resnum[i]+4]=0;//KM
				// errat[i+4]=0;//KM
				flag=0;//reset for next line
				//}//5
			}//4	single atom line end

		}//3	pdb file end

		//cout << "ATOM NUMBER:	"<<atmnum <<endl;
		//cout << "RESNUM[TOTAL]	"<<resnum[atmnum]<<endl;

		//fin.close();

	///DO THE CALCULATION ON THIS PDB FILE DATASET////////

	//STAGE 2: MIN/MAX determination
	for (i=1; i<=3; i++) {min[i]=999; max[i]=-999;}//change to appropriate format
	for (i=1; i<=atmnum; i++)
	{
		for (j=1; j<=3; j++)
		{
			if (xyz[j-1][i]<min[j]) {min[j]=xyz[j-1][i];}
			if (xyz[j-1][i]>max[j]) {max[j]=xyz[j-1][i];}
		}
	}

	fout<<"MINIMUM SPACE LIMITS:	";
	for (j=1;j<=3;j++) {fout <<min[j]<<"	";}
	fout<<endl<<"MAXIMUM SPACE LIMITS:	";
	for (j=1;j<=3;j++) {fout <<max[j]<<"	";}
	fout << endl;


	//STAGE 3: ASSIGN BOXES TO EACH ATOM. TEST/ID 4 OVERFILLED.

	most=0; // most atoms per box in this *.PDB .

	for (i=1; i<=3; i++) //box # in 3-D;
	{nbx[i]=( ( max[i]-min[i] )/boxsize)+1;}//START WITH 1 MINIMUM
	//fout << "NUMBER OF BOXES:"<<endl;
	//fout << "X	"<<nbx[1]<<endl;
	//fout << "Y	"<<nbx[2]<<endl;
	//fout << "Z	"<<nbx[3]<<endl;
	//fout << "TOTAL BOXES:	" << nbx[1]*nbx[2]*nbx[3]<<endl;

//	if ((nbx[1]*nbx[2]*nbx[3])> (bxmx-1))
//	{	fout << "ERROR: TOO MANY BOXES"<<endl; flag2=1;}
    const int bxmx = (nbx[1]*nbx[2]*nbx[3]) + 1;
//    const int bxmx = atmnum + 1;
    int ibox1[16][bxmx];
	if (flag2!=1){//3 flag2 ignores a pdb with too many boxes;
        for (i=1; i<=nbx[1]*nbx[2]*nbx[3]; i++){//declare box holder
        	ibox1[0][i]=0;// resets # of atoms per box
        }
        // Count the number of atoms in each box
        for	(i=1; i<=atmnum; i++){//Translate all atoms so that the minimum is centered on the origin
        	ix=((xyz[0][i]-(min[1]-0.00001) )/boxsize);//0.00001 ensures that all atoms fit into designated boxes
            iy=((xyz[1][i]-(min[2]-0.00001) )/boxsize);
            iz=((xyz[2][i]-(min[3]-0.00001) )/boxsize);
            ind = 1 + ix + iy*nbx[1] + iz*nbx[1]*nbx[2];//box index

            ibox1[0][ind]=ibox1[0][ind] + 1;
            temp =ibox1[0][ind]; // necessary to get into array
            if (temp < 16){
                ibox1[temp][ind]=i;// each atom in that box is listed
            }
            /*//TEST
            fout <<ix<<"	"<<iy<<"	"<<iz<<endl;
                {
                    fout << "ibox1["<<temp<<"]["<<ind<<"]	"<<ibox1[temp][ind] << endl;
                }*/
        }

        //TODAY
        /*for (i=1; i<=nbx[1]*nbx[2]*nbx[3]; i++)//output test
        {
            //if (ibox1[0][i]!=0)
                //fout << "Ind:	"<<i<<"	Atom#:	"<<ibox1[0][i]<< endl;
            for (j=1; j<=ibox1[0][i];j++)
                {
                    fout << "ibox1["<<j<<"]["<<i<<"]	"<<ibox1[j][i] <<" = "<< name[ibox1[j][i]]<< endl;
                }
        }
        */

        for (i=1; i<=nbx[1]*nbx[2]*nbx[3]; i++){
            if (ibox1[0][i]>15){
                fout << "TOO MANY ATOMS IN BOX #"<< i << ":	"<< ibox1[0][i]<< endl;
                for (j=1; (j<=ibox1[0][i])&&(ibox1[0][i]<16);j++){
                    fout << "ibox1["<<j<<"]["<<i<<"]	"<<ibox1[j][i] << endl;
                }
                flag2=1;
            }
            if (ibox1[0][i]> most){ most= ibox1[0][i];}
        }
	//fout <<"Most atoms in a box:	" <<  most << endl;

	}//3 end first flag2

	//STEP 4: PREFORM THE ATOM COMPARISON CALCULATIONS.
	//CONSIDER PUTTING SOME VARIABLES INTO THE TOP OF THE PROGRAM FOR MULTI-FILE DECLARATIONS.

	//count =0;cnt total
	pstat = 0;
	stat = 0;
	mtrxstat=0;
    //bool new_chain = false;
    int last_chain_length = 0;
    //int obs_chain = 1;
    //int residue = 0;
    int residue_counter = 0;
	//NEED A 9 FRAME WINDOW TESTER HERE/ AND FULL STATISTIC OUTPUT AT THE BACK - SIMPLE!
	if (flag2 != 1){//3
	    //if (resnum[1] > 1){//Check when the first residue is not 1, but some other number, say 3 add errat entries
        //      cout << "Found start residue greater than 1 -> " << resnum[1] << " adding " << resnum[1] - 1 << " errat observations" << endl;
        //    for(int missing_start = 1; missing_start <= resnum[1] - 1; missing_start++){
	    //        errat.push_back(0);
	    //    }
	    //}
        for (i=1; i<=atmnum; i++){//4 //throws in all atmnum's
            //fout << i << endl;
            // ensure the measurement happens when a new residue is iterated
            if (((resnum[i] > resnum[i - 1]) || (i==1))/*&&(chain==chainID[i])*/){//5 //gate let's first atom of res through
                //fout << resnum[i] << " is greater than " << resnum[i - 1] << endl;//remove later
                residue_counter++;
                s=1;//resets a counting clock, start at 1.
                for (v = i; ((s < 10) && (v <= atmnum)); v++){//sets frame to 1, go until 9 - to ensure last residue is complete
                    if (resnum[v + 1] > resnum[v]){ // the residue number is not the same
                        if (resnum[v + 1] - resnum[v] < 100){s++;} // ensure they are on the same chain
                        else if (s == 9){s++;}// 9 residues are found and we have incremented chain. Increment s and move on
                        else{
                            //new_chain = true;
                            break;
                        }
                    }
                    else if (v == atmnum){s++;}
                }
                v--;//always sets v back into the frame of the window, counter last v++

                if ((resSeq[v] > resSeq[i]) && (s == 10)){//6 //test for same chain (LIMIT CHAIN TO 1K RES) and completeness of window
                    //if (new_chain){
                    //    obs_chain++;
                    //    last_chain_length = last_chain_length + residue_counter;
                    //    residue_counter = 1;// since new chain is found start counter over
                    //    new_chain = false;
                    //}
                    for (aa = 0; aa < 4; aa++){
                        for (ab = 0; ab < 4; ab++)
                            c[aa][ab] = 0;//sets the count of atom distances to 0 for each combination
                    }
                    //c[first atom][second atom] = length
                    //temporary function that records the # of different interaction
                    //types in the frame (9 residues).
                    //fout <<"i:	"<<i<<"/"<< resnum[i] <<"	v	"<<v<<"/"<<resnum[v]<<endl;
                    for (rer = i; rer <= v; rer++){//7 // v is last atom frame and i(rer) is the first
                        jbx = ((xyz[0][rer] - (min[1] - 0.00001)) / boxsize);//use an additional test when the last v is atmnum
                        jby = ((xyz[1][rer] - (min[2] - 0.00001)) / boxsize);
                        jbz = ((xyz[2][rer] - (min[3] - 0.00001)) / boxsize);
                        //copy the top calculations
                        //set +/- limit on the values of the coordinates box index
                        ibx1 = jbx - ndelta;
                        if (ibx1 < 0) ibx1 = 0;
                        ibx2 = jbx + ndelta;
                        if (ibx2 > nbx[1] - 1) ibx2 = nbx[1] - 1;

                        iby1 = jby - ndelta;
                        if (iby1 < 0) iby1 = 0;
                        iby2 = jby + ndelta;
                        if (iby2 > nbx[2] - 1) iby2 = nbx[2] - 1;

                        ibz1 = jbz - ndelta;
                        if (ibz1<0) ibz1=0;
                        ibz2 = jbz + ndelta;
                        if (ibz2 > nbx[3] - 1) ibz2 = nbx[3] - 1;
                        //fout << rer << endl;
                        //fout <<"JBX:	"<<jbx<<"	"<<"JBY:	"
                        //	 <<jby<<"	"<<"JBZ:	"<<jbz<<endl;

                        //fout <<"IBZ1:	"<< ibz1<<"	IBZ2:	"<<ibz2<<endl;
                        //fout <<"IBY1:	"<< iby1<<"	IBY2:	"<<iby2<<endl;
                        //fout <<"IBX1:	"<< ibx1<<"	IBX2:	"<<ibx2<<endl;
                        //fout << endl;

                        for (j = ibz1; j <= ibz2; j++){//8
                            for (k = iby1; k <= iby2; k++){//9
                                for (l = ibx1; l <= ibx2; l++){//10
                                    ind = 1 + l + k * nbx[1] + j * nbx[1] * nbx[2];//KM 1 gets to one-index, l, k, j terms get the array box position
                                    //fout << "IND:	"<< ind <<"	:	"<< l <<"	"<<k<<"	"<<j<<endl;
                                    for (m = 1; m <= ibox1[0][ind]; m++){//11
                                        n = ibox1[m][ind];// the atomnum index of the interaction box ind(ices) m(th) atom
                                        //fout <<ind<<" "<< m << endl;

                                        if(resnum[rer] != resnum[n]){//12 //residue inequality test
                                            // Find the distance squared between atom idx n and atom idx rer
                                            dsq = 0;
                                            for (p = 0; p <= 2; p++){
                                                dsq = dsq + (pow((xyz[p][n] - xyz[p][rer]), 2));
                                            }
                                            // Check if distance squared is less than limiting radius squared
                                            if (dsq < rsq){//13 //LIMITS - 3.25 to 3.75. //KM rsq = 14.0625
                                                // check if interaction is novel, i.e. non-bonded interactions
                                                if ((bnam[rer] == 1) && (bnam[n] == 1) && // both the atom indices are Nitrogen or Carbon and
                                                    (// the distance is the result of a peptide bond
                                                     ((resnum[n] == resnum[rer] + 1) && (name[rer] == 1) && (name[n] == 2)) ||//only work for n-N and rer-C
                                                     ((resnum[rer] == resnum[n] + 1) && (name[rer] == 2) && (name[n] == 1)
                                                     /*&& (resnum[rer]==resnum[i])*/) //complete symmetry in interaction eval
                                                    ))
                                                {// do nothing
                                                    //fout <<"QQQQQDSQ	"<< dsq <<"	from	"<< rer <<"/"<<resnum[rer]<<"/"<<name[rer]
                                                    //	<<"	to	"<<n<<"/"<<resnum[n]<<"/"<<name[n]<<"	in	"<< sqrt(dsq)<< endl;
                                                    //fout<<xyz[0][n]<<"	"<<xyz[1][n]<<"	"<<xyz[2][n]<<endl;
                                                    //fout<<xyz[0][rer]<<"	"<<xyz[1][rer]<<"	"<<xyz[2][rer]<<endl;
                                                }//some concern with regard - looks line only the first var imp
                                                else{//14
                                                    if ((n >= i) && (n <= v)){//i and v are included in frame
                                                        if (resnum[rer] > resnum[n]){//KM only look at interactions with prior residues on the chain
                                                            if (dsq <= ssq){ //KM ssq=10.5675
                                                                temp1 = 1;
                                                            }
                                                            else{//KM add 2 * the difference in the max distance and the measured distance
                                                                temp1 = 2 * (3.75 - sqrt(dsq));
                                                                //fout<<xyz[0][n]<<"	"<<xyz[1][n]<<"	"<<xyz[2][n]<<endl;
                                                                //fout<<xyz[0][rer]<<"	"<<xyz[1][rer]<<"	"<<xyz[2][rer]<<endl;
                                                            }
                                                            count++;
                                                            c[name[rer]][name[n]]=c[name[rer]][name[n]]+temp1;
                                                            //fout <<"1DSQ	"<< dsq <<"	from	"<< rer <<"/"<<resnum[rer]<<"/"<<name[rer]<<"	to	"
                                                            // <<n<<"/"<<resnum[n]<<"/"<<name[n] <<"	in	"<< sqrt(dsq)<<"/"<<temp1 << endl;
                                                        }
                                                    }
                                                    else{ //KM the atom of interest is outside of the frame. We still add?
                                                        if (dsq <= ssq){//redundant code, but saves time
                                                            temp1 = 1;
                                                        }
                                                        else{
                                                                temp1 = 2 * (3.75 - sqrt(dsq));
                                                        }
                                                        count++;
                                                        c[name[rer]][name[n]]=c[name[rer]][name[n]]+temp1;
                                                        //fout <<"2DSQ	"<< dsq <<"	from	"<< rer <<"/"<<resnum[rer]<<"/"<<name[rer]<<"	to	"
                                                        //	 <<n<<"/"<<resnum[n]<<"/"<<name[n] <<"	in	"<< sqrt(dsq)<<"/"<<temp1 << endl;
                                                    }
                                                }//14
                                            }//13
                                        }//12
                                    }//11
                                }//10
                            }//9
                        }//8
                    }//7

                    temp2 = 0;//total interaction weight measured
                    for (q = 1; q <= 3; q++){
                        for (r = 1; r <= 3; r++){
                            temp2 = temp2 + c[q][r];
                        }
                    }
                    if (temp2>0){//minimum interactions test
                        //fout <<temp2<<" residue:"<<resnum[i]+4<<" count:"<<count<<" (1):"<<c[1][1]/temp2<<" (2):"<<(c[1][2]+c[2][1])/temp2<<" "
                        //<<(c[1][3]+c[3][1])/temp2<<" "<<c[2][2]/temp2<<" "<<(c[2][3]+c[3][2])/temp2<<" "
                        //<<c[3][3]/temp2<<endl;
                    }
                    else{
                        fout <<temp2<<" "<<resnum[i]+4<<" "<<count<<" "<<"WARNING: No Interactions in This Frame"<<endl;
                    }
                    ////Check for gaps in residue numbering and add errat observations accordingly
                    //if ((resnum[i] - resnum[i - 1] > 1) && (chainID[i] == chainID[i - 1])){
                    //    //cout << "found missing residues" << resnum[i - 1] << " to " << resnum[i] << " adding "
                    //    //<< resnum[i] - resnum[i - 1] - 1 << " errat observations" << endl;
                    //    for (int missing_residue=1; missing_residue < resnum[i] - resnum[i - 1]; missing_residue++){
                    //        errat.push_back(0);//add a blank measurement to vector for each residue that is unavailable
                    //    }
                    //}

                    if (temp2 > maxwin){//minimum interactions test
                        //zout << resnum[i]+4 <<"	"<<c[1][1]/temp2<<"	"<<(c[1][2]+c[2][1])/temp2<<"	"
                        //	 <<(c[1][3]+c[3][1])/temp2<<"	"<<c[2][2]/temp2<<"	"
                        //	 <<(c[2][3]+c[3][2])/temp2<<endl;
                        matrix[1] = c[1][1]/temp2;
                        matrix[2] = (c[1][2]+c[2][1])/temp2;
                        matrix[3] = (c[1][3]+c[3][1])/temp2;
                        matrix[4] = c[2][2]/temp2;
                        matrix[5] = (c[2][3]+c[3][2])/temp2;

                        mtrx = matrixdb(matrix);
                        stat++;
                        mtrxstat = mtrxstat + mtrx;
                        mstat = 0;

                        if  (mtrx > lmt[1]){
                            mstat= 99;
                            pstat++;
                            //fout<< "pstat99 "<<resnum[i]+4<<" "<<i<<endl;
                        }
                        else if (mtrx > lmt[2]){
                            mstat= 95;
                            pstat++;
                            //fout<< "pstat95 "<<resnum[i]+4<<" "<<i<<endl;
                        }

                        //fout << resnum[i]+4<<"	"<< mtrx <<"	"<< mstat <<"% errat array #"<< errat.size() << endl;
                        //tyout << resnum[i]+4<<"	"<< mtrx <<"	"<< mstat <<"%"<< endl;

                        //POSTSCRIPT
                        //chainx= (1 + (( resnum[i] - 4 ) / 10000 ));//chain in here
                        // errat[resnum[i]+4]=mtrx;//KM
                        // errat[i+4]=mtrx;//KM
                        //errat.push_back(mtrx);//KM using a pure incremental approach to the errat array
                        //residue = resnum[i] + 4 - ((obs_chain - 1) * chaindif) + last_chain_length;
                        //cout << "Setting residue " << residue << " from atom record " << i << " found by resnum " << resnum[i] << endl;
                        //cout << "Setting with residue counter " << residue_counter << endl;
                        //errat[residue] = mtrx;//KM
                        errat[residue_counter] = mtrx;//KM
                        //KM can't use resSeq as it increments only when residues do, doesn't respect chains
                        //cout << "errat at residue "<< residue_counter + 4 << " is " << mtrx << endl;
                    }
                    else{
                        //errat.push_back(0);
                        lowframe++;
                        fout << "WARNING: Frame	"<< resnum[i] + 4<<"	Below Minimum Interaction Limit."<<endl;
                        //fout << "Low Frames:"<<lowframe << endl;
                    }

                }//6 END of the proper frame test
                else{
                //cout << "Incorrect frame at residue counter " << residue_counter << endl;
                //cout<<"incorrect frame found at residue"<< resnum[i]<<endl;
                //errat.push_back(0);//add a blank measurement to vector as the frame is unavailable
                }
            }//5
        }//4
	}//3 flag2 pdb exclusion end
	if (stat > 0){
        for (i = 1; i <= 4; i++){
            cout << "Residue	" << i << "	" << 0 << endl;
        }
        for (i = 1; i <= residue_counter - 4; i++){
            cout << "Residue	" << i + 4 << "	" << errat[i] << endl;// This is a special spacing character ->"	"
        }
        cout << "Overall quality factor: " << 100 - (100 * (pstat / stat)) << endl;
    }

	//if (stat>0){
    //    zout << "EOF: "<<file<<endl;
    //    zout <<"Total frames: "<<stat<<"	P frames "<<pstat<<"	Number: "<<pstat/stat<<endl<<endl;
    //    zout << "Avg Probability	"<< mtrxstat/stat << endl;
    //    mout << o <<"	"<<file<<"	"<<pstat/stat<<"	"<<mtrxstat/stat<< endl;
    //    //fout << "pstat "<< pstat<< " stat "<<stat<<" pstat/stat "<<pstat/stat<<endl;
    //    //fout << 100-(100*pstat/stat)<<endl;
//
    //    //POSTSCRIPT
    //    chainx = (1 + (resnum[atmnum] - 4) / 10000);//total chains
    //    //cout <<"atmnum, resnum[atmnum], chainx "<< atmnum << "  " << resnum[atmnum] << "  " << chainx<< endl;
//
    //    //z1 controls atmnum
    //    //z2 contros chain array
//
    //    z2=1;//start with 1
    //    //ir1[0]=0;
    //    //ir2[0]=0;
    //    ir1[z2] = resnum[1] + 4 - ((z2 - 1) * chaindif);
    //    //ir2[z2] = 0;// Array with the last residue number in each incremental chain
    //    id_by_chain[z2] = chainID[1];
    //    //cout << "atn, chain#, chainID " << "1" << "  " << z2 << "  " << id_by_chain[z2]<<endl;
    //    last_chain_length = 0;
    //    // find the residues at which the chain transitions
    //    for (z1 = 1 ; z1 < atmnum; z1++){
    //        if (z1 == (atmnum - 1)){//last atom
    //            ir2[z2] = resnum[atmnum] - 4 - ((z2-1) * chaindif) + last_chain_length;
    //        }
    //        else if ((chainID[z1] != chainID[z1 + 1]) && (resnum[z1] > 4)){//ensure no seg problems
    //            // ir2[z2]=resnum[z1]-4;
    //            ir2[z2] = resnum[z1] - 4 - ((z2-1) * chaindif) + last_chain_length;//KM
    //            last_chain_length = last_chain_length + ir2[z2] - ir1[z2] + 9;//KM
    //            // ir2[z2]=resnum[z1]-((z2-1)*chaindif);//Probably need to get rid of the -4 offset as it is now indexed by addition instead of array
//
    //            //cout <<"ir2  "<< ir2[z2]<<"	ir1	"<<ir1[z2]<<endl;
    //            z2++;
    //            ir1[z2] = resnum[z1 + 1] + 4 - ((z2-1) * chaindif) + last_chain_length;//KM
//
    //            id_by_chain[z2] = chainID[z1 + 1];
    //            //cout << "atn, chain#, chainID " << z1 << "  " << z2 << "  " << id_by_chain[z2]<<endl;
    //            //cout <<"z2	"<< z2 <<"	z1	"<<z1 <<"	chainid
    //            //"<<chainID[z1]<<"	chainid+1	"<<chainID[z1+1]<<endl;
    //        }
    //    }
    //    //cout <<"z2 "<<z2 <<"	z1	"<<z1 <<"	x	"<<xyz [0][z1]<<"	resnum
    //    //"<<resnum[z1]<<"	maxres	"<<resnum[atmnum]<<endl;
    //    //cout << "eol"<< endl;
//
    //    mst=0;
//
    //    for (ich=1; ich<=chainx; ich++){
    //        ms = ( (double(ir2[ich] - ir1[ich] + 1))/(300 + 1) );// # pages
    //        //cout <<"# pages	"<<ms << endl;
    //        ms = double(ir2[ich] - ir1[ich] + 1)/(ms);// # residues per page
    //        //cout <<"res per page"	<< ms << endl;
    //        if (ms>mst) mst=ms;
    //        if (mst<200) mst=200;
    //    }
    //    //cout <<"mst		"<< mst << endl;
    //    sz = 200/mst;
    //    cout << "Size of errat array " << errat.size() << endl;
//
    //    for (ich=1; ich<=chainx; ich++){
    //        np = 1 + ((ir2[ich]-ir1[ich]+1)/mst);
    //        //cout <<"np		"<<np << endl;
    //        for (z1 = 1; z1 <= np ; z1++){
    //            ir0 = ir1[ich] + mst * (z1 - 1);
    //            ir = ir0 + mst - 1;
    //            if (ir > ir2[ich]) ir = ir2[ich];
    //            //fout <<"chain "<<ich<<":    Residue range "<< ir0<<" to "<< ir << endl;
    //            fout << "chain " << id_by_chain[ich] << ":    Residue range " << ir0 << " to " << ir << endl;
//
    //            {//PS START HERE
    //            err << "%!PS"<<endl;
    //            err << "%FIXED"<<endl;
    //            err << "/sce {8} def /scr {3} def"<<endl;
    //            err << "90 rotate 110 -380 translate /e95 {11.527} def /e99 {17.191} def"<<endl;
    //            err << "/Helvetica findfont 18 scalefont setfont 0.5 setlinewidth"<<endl;
    //            // err << "/g1 {1} def /g2 {0.6} def /g3 {0.0} def"<<endl;
    //            err << "/bar1 {/g {1 1 1} def bar} def /bar2 {/g {1 1 0} def bar} def"<<endl;
    //            err << "/bar3 {/g {1 0 0} def bar} def /bar {sce mul /yval exch def"<<endl;
    //            err << " scr mul /xval exch def"<<endl;
    //            err << "newpath xval 0 moveto xval yval lineto scr -1 mul 0"<<endl;
    //            err << " rlineto 0 yval -1 mul rlineto closepath gsave g setrgbcolor"<<endl;
    //            err << " fill grestore stroke} def"<<endl;
    //            err << "/tick {newpath 0.5 sub scr mul 0 moveto 0 -3 rlineto"<<endl;
    //            err << " currentpoint stroke moveto -10 -12 rmoveto} def"<<endl;
//
    //            err <<"% VARIABLE"<<endl;
    //            err <<sz<<"   "<<sz<<" scale /rlim {"<< ir-ir0+1 <<"} def"<<endl;
    //            err << "gsave 0 30 sce mul 20 add translate "<<endl;
    //            //err << "0 30 moveto (Chain#:"<<  ich<<") show "<<endl;
    //            err << "0 30 moveto (Chain#:"<<  id_by_chain[ich] <<") show "<<endl;
    //            err << "0 50 moveto (File: "<< file <<") show "<<endl;
    //            err << "0 10 moveto (Overall quality factor**: "
    //                <<  100-(100*pstat/stat)<<")show"<<endl;
    //            err << "0 70 moveto (Program: ERRAT2) show"<<endl;
    //            err << "() show"<<endl;
//
//
    //            err << "% FIXED"<<endl;
    //            err << "grestore newpath 0 0 moveto 0 27 sce mul rlineto stroke"<<endl;//side bars
    //            err << "newpath rlim scr mul 0 moveto 0 27 sce mul rlineto stroke"<<endl;//side bars
    //            err << "newpath 0  0 moveto rlim scr mul 0 rlineto stroke"<<endl;
    //            err << "newpath -3 e95 sce mul moveto rlim scr mul 3 add 0 rlineto"<<endl;
    //            err << "stroke newpath -3 e99 sce mul moveto rlim scr mul 3 add 0"<<endl;
    //            err << " rlineto stroke"<<endl;
    //            err << "newpath 0  27  sce mul moveto rlim scr"<<endl;//top bar
    //            err << " mul 0 rlineto stroke"<<endl;
    //            err << "rlim scr mul 2 div 100 sub -34"<<endl;
    //            err << " moveto (Residue # (window center)) show"<<endl;
    //            err << "/Helvetica findfont 14 scalefont setfont 0.5 setlinewidth"<<endl;
    //            err << "-34 e95 sce mul 4 sub moveto (95\\%) show"<<endl;
    //            err << "-34 e99 sce mul 4 sub moveto (99\\%) show"<<endl;
    //            err << "/Helvetica findfont 12 scalefont setfont 0.5 setlinewidth"<<endl;
    //            err << "0 -70 moveto (*On the error axis, two lines are drawn to indicate the confidence with) show"<<endl;
    //            err << "0 -82 moveto (which it is possible to reject regions that exceed that error value.) show"<<endl;
    //            err << "0 -100 moveto (**Expressed as the percentage of the protein for which the calculated) show"<<endl;
    //            err << "0 -112 moveto (error value falls below the 95\\% rejection limit.  Good high resolution) show"<<endl;
    //            err << "0 -124 moveto (structures generally produce values around 95\\% or higher.  For lower) show"<<endl;
    //            err << "0 -136 moveto (resolutions (2.5 to 3A) the average overall quality factor is around 91\\%. ) show"<<endl;
    //            err << "/Helvetica findfont 18 scalefont setfont 0.5 setlinewidth"<<endl;
    //            err << "gsave -40 -5 translate 90 rotate 80 0 moveto (Error value*)"<<endl;
    //            err << "show grestore"<<endl;
    //            err << "/Helvetica findfont 16 scalefont setfont 0.5 setlinewidth"<<endl;
    //            int chain_length = ir - ir0 + 1;
    //            //cout << chain_length << endl;
    //            for (z2=4; z2<=chain_length; z2++){
    //                if (z2%20==0){
    //                    err << (z2 - 4)    <<" tick        "<<endl;
    //                    //err <<"("<< (z2 - 10000*(z2/10000)	)<<") show	"<< endl; }//KM
    //                    err <<"("<< z2<<") show	"<< endl;
    //                }//KM
    //                else if (z2%10==0){
    //                err << (z2 - 4)	<<" tick	"<<endl;
    //                }
    //            }
    //            for (z2=ir0; z2<=ir; z2++){
    //                strcpy(bar, "bar1\0");
    //                if (errat[z2]>lmt[2]) strcpy(bar,"bar2\0");
    //                if (errat[z2]>lmt[1]) strcpy(bar,"bar3\0");
    //                if (errat[z2]>27) errat[z2]=27;
    //                //fout <<"errt "<< "	z2	"<<z2<<bar<<endl;
    //                err << z2-ir0+1<<"	"<<errat[z2]<<" "<<bar<<endl;
    //            }
    //            err << "showpage" << endl;
    //            }
    //        }
    //    }
//
//
//
    //    /*double errat[size];
    //    int ich;
    //    int ir1[100];
    //    int ir2[100];
    //    double ms, mst;
    //    double sz;
    //    int np, ip, ir0, ir;
    //    int z1, z2;
    //    char bar[5]*/
//
	//}
	//else{
	//	zout <<"Not enough interactions to get data"<<endl;
	//	mout <<o<<"	"<<file<<" Not enough interactions to get data"<<endl;
	//}

	}//2.5 filename

	/////////////////FILENAME LOOP BORDER/////////////////////
	}//2 PDBLLIST LOOP END
//	count_in.close();

	//cin >> n;//keep exe open

	mout.close();
	fout.close();
	zout.close();
	//tyout.close();
	//err.close();
	return 0;
}//1

double matrixdb(double matrix[6])
{//1
	int o=0, u, v;

	double c1[6][6];
	double d1[6][6];
	for (u=1; u<6; u++)//initailize to 0
	{
		for (v=1; v<6; v++)
		{
			c1[u][v]=0;
			d1[u][v]=0;
		}
	}
	/*double b1[6][6] = {	0,0,0,0,0,0,
						0,0,0,0,0,0,
						0,0,0,0,0,0,
						0,0,0,0,0,0,
						0,0,0,0,0,0,
						0,0,0,0,0,0};
	*/
	double avg[6];
	ifstream stat;
	/*stat.open("mtrx.txt");
	if (stat.fail())
	{	cout << "Failed opening mtrx.txt"<< endl;
		exit (1);}
	for (u = 1; u<6; u++)
	{
		for (v = 1; v<6; v++)
		{
			stat >> b1[u][v];
			//fout << b1[u][v]<<"	";
		}//fout << endl;
	}//fout << endl;
	stat.close();
	*/
	double b1[6][6] = {	0,0,0,0,0,0,
						0,	5040.279078850848200,	3408.805141583649400,	4152.904423767300600,	4236.200004171890200,	5054.781210204625500,
0,	3408.805141583648900,	8491.906094010220800,	5958.881777877950300,	1521.387352718486200,	4304.078200827221700,
0,	4152.904423767301500,	5958.881777877952100,	7637.167089335050100,	6620.715738223072500,	5287.691183798410700,
0,	4236.200004171890200,	1521.387352718486200,	6620.715738223072500,	18368.343774298410000,	4050.797811118806700,
0,	5054.781210204625500,	4304.078200827220800,	5287.691183798409800,	4050.797811118806700,	6666.856740479164700};

	/*
	stat.open("avg.txt");
	if (stat.fail())
	{	cout << "Failed opening avg.txt"<< endl;
		exit (1);}
		for (u = 1; u<6; u++)
		{
			stat >> avg[u];
			//fout << avg[u]<<endl;
		}
	stat.close();
	*/
	avg[1]=0.192765509919262;
	avg[2]=0.195575208778518;
	avg[3]=0.275322406824210;
	avg[4]=0.059102357035642;
	avg[5]=0.233154192767480;

	//Multiplication Stage
	int m1,n1,k1,p1;
	double a[6][6],b[6][6],x;//redefinition

	//STREAMS
	ofstream fout;// output logs to files


	//Begins Processing
	fout << "PROCESSING FRAME STATISTICS:"<< endl;

		for (u = 1; u<6; u++)
		{
			matrix[u]=matrix[u]- avg[u];
		}

			//Multiplication Stage: input temp, output c
			{
					fout<<"Vertical Matrix A";
					m1=1;//Number of rows
					p1=5;//Number of columns

					for (u=1;u<=m1;u++)
						{
						for(v=1;v<=p1;v++)
							{
							a[u][v]=matrix[v];
							b[v][u]=matrix[v];
							}
						 }
					for (u=1;u<=m1;u++)
						{
						fout <<endl;
						for(v=1;v<=p1;v++)
							{
							fout<<a[u][v]<<"	";
							}
						}

					p1=5;//Number of Rows
					n1=5;//Number of columns

					//MATRIX PRODUCT AxB1 = C1 //
					for (u=1;u<=m1;u++)
						{
						for(v=1;v<=n1;v++)
							{
							x=0;
							for (k1=1;k1<=p1;k1++)
								{
								x=x+a[u][k1]*b1[k1][v];
								}
							c1[u][v]=x;
							}
						}

					fout << endl;
					fout<<"Matrix Product AxB1 = C1 "<<m1<<" x "<<n1;
					for (u=1;u<=m1;u++)
						{
						fout <<endl;
						for(v=1;v<=n1;v++)
							{
							fout<<c1[u][v]<<"	";
							}
						}
					}
///////////

			fout << endl;
					fout << "Horizontal Matrix B";
					p1=5;//Number of Rows
					n1=1;//Number of columns
					for (u=1;u<=p1;u++)
					{
						fout <<endl;
						for(v=1;v<=n1;v++)
						{
							fout<<b[u][v]<<"	";
						}
					}


					//MATRIX PRODUCT d1 //
					for (u=1;u<=m1;u++)
						{
						for(v=1;v<=n1;v++)
							{
							x=0;
							for (k1=1;k1<=p1;k1++)
								{
								x=x+c1[u][k1]*b[k1][v];
								}
							d1[u][v]=x;
							}
						}

					fout << endl<< "Total Matrix "<<endl;

							fout << d1[1][1]<<"	";

					fout << endl;

	fout.close();

	return d1[1][1];
}//1