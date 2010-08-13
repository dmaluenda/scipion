/***************************************************************************
 *
 * Authors:     Carlos Oscar S. Sorzano (coss@cnb.csic.es)
 *
 * Unidad de  Bioinformatica of Centro Nacional de Biotecnologia , CSIC
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA
 * 02111-1307  USA
 *
 *  All comments concerning this program package may be sent to the
 *  e-mail address 'xmipp@cnb.csic.es'
 ***************************************************************************/

#ifndef _DOWNSAMPLE
#define _DOWNSAMPLE

#include <data/funcs.h>
#include <data/micrograph.h>

///@defgroup MicrographDownsample Micrograph Downsample
/// @ingroup ReconsLibrary
//@{
/** Downsample parameters. */
class Prog_downsample_prm
{
public:
    /// Input micrograph
    FileName fn_micrograph;

    /// Output micrograph
    FileName fn_downsampled;

    /// Xstep
    int      Xstep;
    /// Ystep
    int      Ystep;
    /// Scale. Alternative way to give output dimensions.
    double scale;

#define KER_RECTANGLE  0
#define KER_CIRCLE     1
#define KER_GAUSSIAN   2
#define KER_PICK       3
#define KER_SINC       4

    /** Kernel mode.
        Valid modes are KER_RECTANGLE, KER_CIRCLE, KER_GAUSSIAN
        KER_PICK and KER_SINC */
    int      kernel_mode;

    /// Circle radius
    double   r;

    /// Gaussian sigma
    double   sigma;

    /// delta
    double   delta;

    /// Deltaw
    double   Deltaw;
    
    /// fourier interpolation
    bool do_fourier;
    /// number of Threads used in the Fourier TRansform
    int nThreads;
    /// Rectangular X size
    int      Xrect;
    /// Rectangular Y size
    int      Yrect;
    /// Output bits
    int      bitsMp;
    /// Reverse endian
    bool     reversed;
public:
    // Side information
    // Kernel
    MultidimArray<double> kernel;
    // Input micrograph
    Micrograph M;
    // Input micrograph depth
    int bitsM;
    // Input dimensions
    int Xdim, Ydim;
    // Output dimensions
    int Xpdim, Ypdim;
public:
    /** Read input parameters.
        If do_not_read_files=TRUE then fn_micrograph and
        fn_downsampled are not read. */
    void read(int argc, char **argv, bool do_not_read_files = false);

    /// Usage
    void usage() const;
#ifdef NEVERDEFINED
    /// Produce command line parameters
    std::string command_line() const;
#endif
    /// Generate kernel
    void generate_kernel();

    /// Open input micrograph
    void open_input_micrograph();

    /// Close input micrograph
    void close_input_micrograph();

    /** Create output information file */
    void create_empty_output_file();

    /** Really downsample.*/
    void Downsample() const;
};
//@}
#endif
