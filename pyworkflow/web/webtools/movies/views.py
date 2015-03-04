# **************************************************************************
# *
# * Authors:    Jose Gutierrez (jose.gutierrez@cnb.csic.es)
# *
# * Unidad de  Bioinformatica of Centro Nacional de Biotecnologia , CSIC
# *
# * This program is free software; you can redistribute it and/or modify
# * it under the terms of the GNU General Public License as published by
# * the Free Software Foundation; either version 2 of the License, or
# * (at your option) any later version.
# *
# * This program is distributed in the hope that it will be useful,
# * but WITHOUT ANY WARRANTY; without even the implied warranty of
# * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# * GNU General Public License for more details.
# *
# * You should have received a copy of the GNU General Public License
# * along with this program; if not, write to the Free Software
# * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA
# * 02111-1307  USA
# *
# *  All comments concerning this program package may be sent to the
# *  e-mail address 'jmdelarosa@cnb.csic.es'
# *
# **************************************************************************

from os.path import exists, join, basename
from pyworkflow.web.app.views_util import loadProject, getResourceCss, getResourceJs
from pyworkflow.web.app.views_base import base_grid, base_flex
from pyworkflow.web.app.views_project import contentContext
from django.shortcuts import render_to_response
from pyworkflow.web.pages import settings as django_settings
from pyworkflow.manager import Manager
from django.http import HttpResponse
from pyworkflow.tests.tests import DataSet
from pyworkflow.utils import copyFile

def service_movies(request):

    if 'projectName' in request.session: request.session['projectName'] = ""
    if 'projectPath' in request.session: request.session['projectPath'] = ""

    movies_utils = django_settings.STATIC_URL + "js/movies_utils.js"

    context = {'projects_css': getResourceCss('projects'),
               'project_utils_js': getResourceJs('project_utils'),
               'movies_utils': movies_utils,
               'hiddenTreeProt': True,
               }
    
    context = base_grid(request, context)
    return render_to_response('movies_projects.html', context)


def writeCustomMenu(customMenu):
    if not exists(customMenu):
        f = open(customMenu, 'w+')
        f.write('''
[PROTOCOLS]

Movies_Alignment = [
    {"tag": "section", "text": "1. Upload data", "children": [
        {"tag": "url", "value": "/upload/", "text": "Upload Data", "icon": "fa-upload.png"}]},
    {"tag": "section", "text": "2. Select your data", "children": [
        {"tag": "protocol", "value": "ProtImportMovies", "text": "Select Movies", "icon": "bookmark.png"}]},
    {"tag": "section", "text": "3. Align your Movies", "children": [
        {"tag": "protocol", "value": "ProtImportMovies", "text": "xmipp3 - movie alignment"}]}]
        ''')
        f.close()
        
def create_movies_project(request):
    
    if request.is_ajax():
        
        import os
        from pyworkflow.object import Pointer
        from pyworkflow.em.protocol import ProtImportMovies
        from pyworkflow.em.packages.xmipp3 import ProtMovieAlignment
        
        # Create a new project
        manager = Manager()
        projectName = request.GET.get('projectName')
        
        # Filename to use as test data 
        testDataKey = request.GET.get('testData')
        
        #customMenu = os.path.join(os.path.dirname(os.environ['SCIPION_PROTOCOLS']), 'menu_initvolume.conf')
        customMenu = os.path.join(os.environ['HOME'], '.config/scipion/menu_movies.conf')
        writeCustomMenu(customMenu)
        
        project = manager.createProject(projectName, runsView=1, protocolsConf=customMenu)   
        
        # 1. Import movies
        protImport = project.newProtocol(ProtImportMovies,
                                         objLabel='select movies')
        project.saveProtocol(protImport)   
        
        # 2. Movie Alignment 
        protMovAlign = project.newProtocol(ProtMovieAlignment)
        protMovAlign.setObjLabel('xmipp - movie alignment')
        protMovAlign.inputMovies.set(protImport)
        protMovAlign.inputMovies.setExtendedAttribute('outputMovies')
        project.saveProtocol(protMovAlign)
        
        """
        # If using test data execute the import averages run
        # options are set in 'project_utils.js'
        dsMDA = DataSet.getDataSet('initial_volume')
        
        if testDataKey :
            fn = dsMDA.getFile(testDataKey)
            newFn = join(project.uploadPath, basename(fn))
            copyFile(fn, newFn)
            
            protImport.filesPath.set(newFn)
            protImport.samplingRate.set(1.)
#             protImport.setObjectLabel('import averages (%s)' % testDataKey)
            
            project.launchProtocol(protImport, wait=True)
        else:
            project.saveProtocol(protImport)
    """
    return HttpResponse(mimetype='application/javascript')

def get_testdata(request):
    # Filename to use as test data 
    testDataKey = request.GET.get('testData')
    dsMDA = DataSet.getDataSet('initial_volume')
    fn = dsMDA.getFile(testDataKey)
    return HttpResponse(fn, mimetype='application/javascript')

def check_m_id(request):
    result = 0
    projectName = request.GET.get('code', None)
    
    try:
        manager = Manager()
        project = loadProject(projectName)
        result = 1
    except Exception:
        pass
    
    return HttpResponse(result, mimetype='application/javascript')
 
def movies_content(request):
    projectName = request.GET.get('p', None)
    path_files = '/resources_movies/img/'
    
    context = contentContext(request, projectName)
    context.update({
                    # MODE
                    'mode':'service',
                    
                    # IMAGES
#                     'imageName': path_files + 'image.png',
                    })
    
    return render_to_response('movies_content.html', context)
