/*

miniMDock is a miniapp of the GPU version of AutoDock 4.2 running a Lamarckian Genetic Algorithm
Copyright (C) 2017 TU Darmstadt, Embedded Systems and Applications Group, Germany. All rights reserved.
For some of the code, Copyright (C) 2019 Computational Structural Biology Center, the Scripps Research Institute.

AutoDock is a Trade Mark of the Scripps Research Institute.

This library is free software; you can redistribute it and/or
modify it under the terms of the GNU Lesser General Public
License as published by the Free Software Foundation; either
version 2.1 of the License, or (at your option) any later version.

This library is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public
License along with this library; if not, write to the Free Software
Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA

*/


#include <stdio.h>
#include <stdlib.h>
#include <vector>

#ifdef USE_KOKKOS
#include <Kokkos_Core.hpp>
#include "kokkos_settings.hpp"
#endif

#include "processgrid.h"
#include "processligand.h"
#include "getparameters.h"
#include "setup.hpp"

#ifdef USE_KOKKOS
int setup(Gridinfo&            mygrid,
	  Kokkos::View<float*,HostType>& floatgrids,
	  Dockpars&            mypars,
	  Liganddata&          myligand_init,
	  Liganddata&          myxrayligand,
	  int i_file,
	  int argc, char* argv[])
{
	//------------------------------------------------------------
	// Capturing names of grid parameter file and ligand pdbqt file
	//------------------------------------------------------------

	// Filling the filename and coeffs fields of mypars according to command line arguments
	if (get_filenames_and_ADcoeffs(&argc, argv, &mypars ) != 0)
		{printf("\n\nError in get_filenames_and_ADcoeffs, stopped job."); return 1;}

	//------------------------------------------------------------
	// Testing command line arguments for cgmaps parameter
	// since we need it at grid creation time
	//------------------------------------------------------------
	mypars.cgmaps = 0; // default is 0 (use one maps for every CGx or Gx atom types, respectively)
	for (unsigned int i=1; i<argc-1; i+=2)
	{
		// ----------------------------------
		//Argument: Use individual maps for CG-G0 instead of the same one
		if (strcmp("-cgmaps", argv [i]) == 0)
		{
			int tempint;
			sscanf(argv [i+1], "%d", &tempint);
			if (tempint == 0)
				mypars.cgmaps = 0;
			else
				mypars.cgmaps = 1;
		}
	}

	//------------------------------------------------------------
	// Processing receptor and ligand files
	//------------------------------------------------------------

	// Filling mygrid according to the gpf file
	if (get_gridinfo(mypars.fldfile, &mygrid) != 0)
		{printf("\n\nError in get_gridinfo, stopped job."); return 1;}

	// Filling the atom types filed of myligand according to the grid types
	if (init_liganddata(mypars.ligandfile, &myligand_init, &mygrid, mypars.cgmaps) != 0)
		{printf("\n\nError in init_liganddata, stopped job."); return 1;}

	// Filling myligand according to the pdbqt file
	if (get_liganddata(mypars.ligandfile, &myligand_init, mypars.coeffs.AD4_coeff_vdW, mypars.coeffs.AD4_coeff_hb) != 0)
		{printf("\n\nError in get_liganddata, stopped job."); return 1;}

	// Resize grid
	Kokkos::resize(floatgrids, 4*(mygrid.num_of_atypes+2)*mygrid.size_xyz[0]*mygrid.size_xyz[1]*mygrid.size_xyz[2]);

	//Reading the grid files and storing values in the memory region pointed by floatgrids
	if (get_gridvalues_f(&mygrid, floatgrids.data(), mypars.cgmaps) != 0)
		{printf("\n\nError in get_gridvalues_f, stopped job."); return 1;}

	//------------------------------------------------------------
	// Capturing algorithm parameters (command line args)
	//------------------------------------------------------------
	get_commandpars(&argc, argv, &(mygrid.spacing), &mypars);

	Gridinfo mydummygrid;
	// if -lxrayfile provided, then read xray ligand data
	if (mypars.given_xrayligandfile == true) {
		if (init_liganddata(mypars.xrayligandfile, &myxrayligand, &mydummygrid, mypars.cgmaps) != 0)
			{printf("\n\nError in init_liganddata, stopped job."); return 1;}

		if (get_liganddata(mypars.xrayligandfile, &myxrayligand, mypars.coeffs.AD4_coeff_vdW, mypars.coeffs.AD4_coeff_hb) != 0)
			{printf("\n\nError in get_liganddata, stopped job."); return 1;}
	}

	//------------------------------------------------------------
	// Calculating energies of reference ligand if required
	//------------------------------------------------------------
/*	if (mypars.reflig_en_reqired == 1) {
		print_ref_lig_energies_f(myligand_init,
					 mypars.smooth,
					 mygrid,
					 floatgrids.data(),
					 mypars.coeffs.scaled_AD4_coeff_elec,
					 mypars.coeffs.AD4_coeff_desolv,
					 mypars.qasp);
	}
*/
	return 0;
}

#else

int setup(std::vector<Map>& all_maps,
	  Gridinfo&            mygrid,
	  std::vector<float>& floatgrids,
	  Dockpars&            mypars,
	  Liganddata&          myligand_init,
	  Liganddata&          myxrayligand,
	  int i_file,
	  int argc, char* argv[])
{
	//------------------------------------------------------------
	// Capturing names of grid parameter file and ligand pdbqt file
	//------------------------------------------------------------


	// Filling the filename and coeffs fields of mypars according to command line arguments
	if (get_filenames_and_ADcoeffs(&argc, argv, &mypars) != 0)
		{printf("\n\nError in get_filenames_and_ADcoeffs, stopped job."); return 1;}

	//------------------------------------------------------------
	// Testing command line arguments for cgmaps parameter
	// since we need it at grid creation time
	//------------------------------------------------------------
	mypars.cgmaps = 0; // default is 0 (use one maps for every CGx or Gx atom types, respectively)
	for (unsigned int i=1; i<argc-1; i+=2)
	{
		// ----------------------------------
		//Argument: Use individual maps for CG-G0 instead of the same one
		if (strcmp("-cgmaps", argv [i]) == 0)
		{
			int tempint;
			sscanf(argv [i+1], "%d", &tempint);
			if (tempint == 0)
				mypars.cgmaps = 0;
			else
				mypars.cgmaps = 1;
		}
	}

	//------------------------------------------------------------
	// Processing receptor and ligand files
	//------------------------------------------------------------

	// Filling mygrid according to the gpf file
	if (get_gridinfo(mypars.fldfile, &mygrid) != 0)
		{printf("\n\nError in get_gridinfo, stopped job."); return 1;}

	// Filling the atom types filed of myligand according to the grid types
	if (init_liganddata(mypars.ligandfile, &myligand_init, &mygrid, mypars.cgmaps) != 0)
		{printf("\n\nError in init_liganddata, stopped job."); return 1;}

	// Filling myligand according to the pdbqt file
	if (get_liganddata(mypars.ligandfile, &myligand_init, mypars.coeffs.AD4_coeff_vdW, mypars.coeffs.AD4_coeff_hb) != 0)
		{printf("\n\nError in get_liganddata, stopped job."); return 1;}

	// Resize grid
	floatgrids.resize(4*(mygrid.num_of_atypes+2)*mygrid.size_xyz[0]*mygrid.size_xyz[1]*mygrid.size_xyz[2]);

	//Reading the grid files and storing values in the memory region pointed by floatgrids
	if (get_gridvalues_f(&mygrid, floatgrids.data(), mypars.cgmaps) != 0)
		{printf("\n\nError in get_gridvalues_f, stopped job."); return 1;}

	//------------------------------------------------------------
	// Capturing algorithm parameters (command line args)
	//------------------------------------------------------------
	get_commandpars(&argc, argv, &(mygrid.spacing), &mypars);


	Gridinfo mydummygrid;
	// if -lxrayfile provided, then read xray ligand data
	if (mypars.given_xrayligandfile == true) {
		if (init_liganddata(mypars.xrayligandfile, &myxrayligand, &mydummygrid, mypars.cgmaps) != 0)
			{printf("\n\nError in init_liganddata, stopped job."); return 1;}

		if (get_liganddata(mypars.xrayligandfile, &myxrayligand, mypars.coeffs.AD4_coeff_vdW, mypars.coeffs.AD4_coeff_hb) != 0)
			{printf("\n\nError in get_liganddata, stopped job."); return 1;}
	}

	//------------------------------------------------------------
	// Calculating energies of reference ligand if required
	//------------------------------------------------------------
	if (mypars.reflig_en_required) {
		print_ref_lig_energies_f(myligand_init,
					 mypars.smooth,
					 mygrid,
					 floatgrids.data(),
					 mypars.coeffs.scaled_AD4_coeff_elec,
					 mypars.coeffs.AD4_coeff_desolv,
					 mypars.qasp);
	}

	return 0;
}

int fill_maplist(const char* fldfilename, std::vector<Map>& all_maps)
{
	std::ifstream file(fldfilename);
        if(file.fail()){
                printf("\nError: Could not open %s. Check path and permissions.",fldfilename);
                return 1;
        }
	std::string line;
	bool prev_line_was_fld=false;
	while(std::getline(file, line)) {
		std::stringstream sline(line.c_str());
		// Split line by spaces:
		std::string word;
		bool is_variable_line=false;
		while(std::getline(sline, word, ' ')){
			// Check if first word is "variable"
			if (word.compare("variable") == 0) is_variable_line=true;
			int len = word.size();
                        if (is_variable_line && len>=4 && word.compare(len-4,4,".map") == 0){ // Found a word that ends in "map"
				// Split the map into segments e.g. protein.O.map -> "protein", "O", "map"
				std::stringstream mapword(word.c_str());
				std::string segment;
				std::vector<std::string> seglist;
				while(std::getline(mapword, segment, '.')) seglist.push_back(segment);

				// Create a new map with the atom name
				all_maps.push_back(Map(seglist[seglist.size()-2]));
			}
		}
	}
	return 0;
}

int load_all_maps (const char* fldfilename, const Gridinfo* mygrid, std::vector<Map>& all_maps, bool cgmaps)
{
	// First, parse .fld file to get map names
	if(fill_maplist(fldfilename,all_maps)==1) return 1;

	// Now fill the maps
        int t, x, y, z;
        FILE* fp;
        char tempstr [128];
	int size_of_one_map = 4*mygrid->size_xyz[0]*mygrid->size_xyz[1]*mygrid->size_xyz[2];

        for (t=0; t < all_maps.size(); t++)
        {
		all_maps[t].grid.resize(size_of_one_map);
		float* mypoi = all_maps[t].grid.data();
                //opening corresponding .map file
                //-------------------------------------
                // Added the complete path of associated grid files.
                strcpy(tempstr,mygrid->grid_file_path);
                strcat(tempstr, "/");
                strcat(tempstr, mygrid->receptor_name);

                //strcpy(tempstr, mygrid->receptor_name);
                //-------------------------------------
                strcat(tempstr, ".");
                strcat(tempstr, all_maps[t].atype.c_str());
                strcat(tempstr, ".map");
                fp = fopen(tempstr, "rb"); // fp = fopen(tempstr, "r");
                if (fp == NULL)
                {
                        printf("Error: can't open %s!\n", tempstr);
                        if ((strncmp(all_maps[t].atype.c_str(),"CG",2)==0) ||
                            (strncmp(all_maps[t].atype.c_str(),"G",1)==0))
                        {
                                if(cgmaps)
                                        printf("-> Expecting an individual map for each CGx and Gx (x=0..9) atom type.\n");
                                else
                                        printf("-> Expecting one map file, ending in .CG.map and .G0.map, for CGx and Gx atom types, respectively.\n");
                        }
                        return 1;
                }

                //seeking to first data
                do    fscanf(fp, "%s", tempstr);
                while (strcmp(tempstr, "CENTER") != 0);
                fscanf(fp, "%s", tempstr);
                fscanf(fp, "%s", tempstr);
                fscanf(fp, "%s", tempstr);

                unsigned int g1 = mygrid->size_xyz[0];
                unsigned int g2 = g1*mygrid->size_xyz[1];
                //reading values
                for (z=0; z < mygrid->size_xyz[2]; z++)
                        for (y=0; y < mygrid->size_xyz[1]; y++)
                                for (x=0; x < mygrid->size_xyz[0]; x++)
                                {
                                        fscanf(fp, "%f", mypoi);
                                        // fill in duplicate data for linearized memory access in kernel
                                        if(y>0) *(mypoi-4*g1+1) = *mypoi;
                                        if(z>0) *(mypoi-4*g2+2) = *mypoi;
                                        if(y>0 && z>0) *(mypoi-4*(g2+g1)+3) = *mypoi;
                                        mypoi+=4;
                                }

		fclose(fp);
	}
        return 0;
}

int copy_from_all_maps (const Gridinfo* mygrid, float* fgrids, std::vector<Map>& all_maps)
{
	int size_of_one_map = 4*mygrid->size_xyz[0]*mygrid->size_xyz[1]*mygrid->size_xyz[2];
        for (int t=0; t < mygrid->num_of_atypes+2; t++) {
		// Look in all_maps for desired map
		int i_map = -1;
		for (int i_atype=0; i_atype < all_maps.size(); i_atype++){
			if (strcmp(mygrid->grid_types[t],all_maps[i_atype].atype.c_str())==0){
				i_map = i_atype; // Found the map!
				break;
			}
		}
		if (i_map == -1){ // Didnt find the map
			printf("\nError: The %s map needed for the ligand was not found in the .fld file!", mygrid->grid_types[t]);
			return 1;
		}

		// Copy from all_maps into fgrids
		memcpy(fgrids+t*size_of_one_map,all_maps[i_map].grid.data(),sizeof(float)*all_maps[i_map].grid.size());
        }

        return 0;
}
#endif
