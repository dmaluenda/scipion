/***************************************************************************
 * Authors:     J.M. de la Rosa Trevin (jmdelarosa@cnb.csic.es)
 *
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

#ifndef PROGRAM_H_
#define PROGRAM_H_

#include "argsprinter.h"
#include "xmipp_error.h"
#include "xmipp_strings.h"
#include "metadata.h"
#include "xmipp_image.h"
#include "xmipp_program_sql.h"

#define XMIPP_MAJOR 3
#define XMIPP_MINOR 0

/** @defgroup Programs2 Basic structure for Xmipp programs
 *  @ingroup DataLibrary
 *
 * General program routines.
 *
 * Here you will find some routines which might help you to write programs
 * with a common structure where only a small part changes. These routines
 * make the heavy work for you and you only have to supply the changing
 * part from one to another.
 *
 * @{
 */

/** This class represent an Xmipp Program.
 * It have some of the basic functionalities of
 * the programs like argument parsing, checking, usage printing.
 */
class XmippProgram
{
private:
    /** Initialization function */
    void init();

    /** Function to check built-ins actions like --more, --help...etc */
    bool checkBuiltIns();

    /** Write Program info to DB */
    void writeToDB();
    /** Print some information to stdout */
    void writeToProtocol();

    /** Create program GUI */
    /** By default, a simple Tk GUI is create based on parameters definition.
     * If an specific program implements a more specialized GUI, should redefine this function.
     */
    virtual void createGUI();
    /** Create Wiki for help */
    void createWiki();

protected:
    /** Define Commons */
    void defineCommons();
    /** Value to store possible error codes */
    int errorCode;
    /// Program definition and arguments parser
    ProgramDef * progDef;

    /** Default comments */
    std::map<String, CommentList> defaultComments;

    ///Original command line arguments
    int argc;
    char ** argv;

public:
    /** Flag to check whether to run or not*/
    bool doRun;
    /** Flag to mark if the program should run without arguments
     * or just print the usage(default behavior)
     */
    bool runWithoutArgs;
    /** @name Functions to be implemented by subclasses.
     * @{
     */
    /** Add the comments for a given default parameter */
    void processDefaultComment(const char *param, const char *left);
    /** Initialize comments for -v, ...*/
    virtual void initComments();
    /** Function in which the param of each Program are defined. */
    virtual void defineParams();
    /** Function in which each program will read parameters that it need.
     * If some error occurs the usage will be printed out.
     * */
    virtual void readParams();
    /** @name Program definitions
     * Following functions will be used in defineParams()
     * for define the arguments of
     * a program. Very useful for checking the command line parameters
     * and for standard usage message.
     * @{
     */
    /** Set the program name */
    void setProgramName(const char * name);
    /** Add usage line */
    void addUsageLine(const char * line, bool verbatim=false);
    /** Clear usage */
    void clearUsage();
    /** Add examples */
    void addExampleLine(const char * example, bool verbatim=true);
    /** Add other programs.
     * Separated by commas and without xmipp_
     */
    void addSeeAlsoLine(const char * seeAlso);
    /** Add keywords to the program definition */
    void addKeywords(const char * keywords);

    /** @} */

    /** Get the argument of this param, first start at 0 position */
    const char * getParam(const char * param, int arg = 0);
    const char * getParam(const char * param, const char * subparam, int arg = 0);
    int getIntParam(const char * param, int arg = 0);
    int getIntParam(const char * param, const char * subparam, int arg = 0);
    double getDoubleParam(const char * param, int arg = 0);
    double getDoubleParam(const char * param, const char * subparam, int arg = 0);
    /** Get arguments supplied to param as a list */
    void getListParam(const char * param, StringVector &list);
    /** Get the number of arguments supplied to the param */
    int getCountParam(const char * param);
    /** Check if the param was supplied to command line */
    bool checkParam(const char * param);
    /** Return true if the program is defined */
    bool existsParam(const char * param);

    /** Add a params definition line*/
    void addParamsLine(const String &line);
    void addParamsLine(const char * line);

    /** Get Parameter definition */
    ParamDef * getParamDef(const char * param) const;

    /// Verbosity level
    int verbose;
    /** debug flag and seed for randomization */
    int debug, seed;

    /** @name Public common functions
     * The functions in this section are available for all programs
     * and should not be reimplemented in derived class, since they
     * are thought to have the same behaivor and it depends on the
     * definition of each program.
     * @{
     */
    /** Returns the name of the program
     * the name of the program is defined
     * by each subclass of this base class.
     */
    const char * name() const;
    /** Print the usage of the program, reduced version */
    virtual void usage(int verb = 0) const;
    /** Print help about specific parameter */
    virtual void usage(const String & param, int verb = 2);

    /** Return the version of the program */
    int version() const;

    /** Show parameters */
    virtual void show() const;

    /** Read the command line arguments
     * If an error occurs while reading arguments,
     * the error message will be printed and the
     * usage of the program showed. So you don't need
     * to do that in readParams();
     * */
    virtual void read(int argc, char ** argv, bool reportErrors = true);
    /** Read arguments from an string.
     * This function should do the same as reading arguments
     * but first convert the string to arguments.
     */
    void read(const String &argumentsLine);

    /** @} */
    /** This function will be start running the program.
     * it also should be implemented by derived classes.
     */
    virtual void run();
    /** function to exit the program
     * can be usefull redefined for mpi programs
     */
    virtual void quit(int exit_code = 0) const;
    /** Call the run function inside a try/catch block
     * The function will return the error code when
     * 0 means success
     * and a value greater than 0 represents the error type
     * */
    virtual int tryRun();

    /** @name Constructors
     * @{
     */
    /** Constructor */
    XmippProgram();
    /** Constructor for read params */
    XmippProgram(int argc, char ** argv);
    /** Destructor */
    virtual ~XmippProgram();
    /** @} */

}
;//end of class XmippProgram

/** Special class of XmippProgram that performs some operation related with processing images.
 * It can receive a file with images(MetaData) or a single image.
 * The function processImage is virtual here and needs to be implemented by derived classes.
 * Optionally can be implemented preProcess and postProcess to perform some customs actions.
 */
class XmippMetadataProgram: public virtual XmippProgram
{
protected:
public:
    //Image<double>   img;
    /// Filenames of input and output Metadata
    FileName        fn_in, fn_out;
    /// Apply geo
    bool apply_geo;
    /// Output dimensions
    int zdimOut, ydimOut, xdimOut;
    size_t ndimOut;
    /// Number of input elements
    size_t mdInSize;

protected:
    /// Metadata writing mode: OVERWRITE, APPEND
    WriteModeMetaData mode;
    /// Input and output metadatas
    MetaData     mdIn, mdOut;
    /// Iterator over input metadata
    MDIterator * iter;
    /// Filenames of input and output Images
    //FileName        fnImg, fnImgOut;
    /// Output extension and root
    FileName oext, oroot;
    /// Set this true to allow the option
    /// of applying the transformation as stored in the header
    bool allow_apply_geo;
    /// Use this flag for not writing at every image and when the program produces an unique output
    bool produces_an_output;
    /// Set this flag true when the program produces a metadata/stack result
    bool each_image_produces_an_output;
    /// Use this flag for not producing a time bar
    bool allow_time_bar;
    /// Some time bar related counters
    size_t time_bar_step, time_bar_size, time_bar_done;
    /// Flag to know when input is a single image or stack
    bool single_image;
    bool input_is_stack, create_empty_stackfile;
    bool delete_output_stack; // Delete previous output stack file
    /// Flag to treat a stack file as a set of images instead of a unique file
    bool decompose_stacks;
    /// Flag to save the output metadata when output file is a stack
    bool save_metadata_stack;
    /// Remove disabled images from the input selfile
    bool remove_disabled;
    /// Object id of the output metadata
    size_t newId;

    virtual void initComments();
    virtual void defineParams();
    virtual void readParams();
    virtual void preProcess();
    virtual void postProcess();
    virtual void processImage(const FileName &fnImg, const FileName &fnImgOut, size_t objId) = 0;
    virtual void show();
    /** Do some stuff before starting processing
     * in a parallel environment usually this only be executed
     * by master.
     */
    virtual void startProcessing();
    virtual void finishProcessing();
    virtual void showProgress();
    /** This method will be used to distribute the images to process
     * it will set the objectId and objectIndexto read from input metadata
     * or -1 if there are no more images to process.
     * This method will be useful for parallel task distribution
     */
    virtual bool getImageToProcess(size_t &objId, size_t &objIndex);

public:
    XmippMetadataProgram();
    void setMode(WriteModeMetaData _mode)
    {
        mode = _mode;
    }
    virtual void run();
}
;// end of class XmippMetadataProgram

/** This class will serve as an interface for python scripts
 * useful for command line parsing and help message printing
 */
class XmippProgramGeneric: public XmippProgram
{
public:
  bool definitionComplete;
   ///Constructor
  XmippProgramGeneric();
  void endDefinition();
  virtual void read(int argc, char ** argv, bool reportErrors = true);

protected:
    void defineParams();
    void readParams();
    void show();
    void run();
}
;// end of class XmippProgramGeneric

/** This macro will be useful for define the main and run
 * an XmippProgram
 */
#define RUN_XMIPP_PROGRAM(progName) \
  int main(int argc, char** argv) { \
      progName program; program.read(argc, argv);\
      return program.tryRun();}


///Declare all programs
/** @} */

#endif /* PROGRAM_H_ */
