# Visual-Humor-Dataset
The Repositary contains code for TSM (Textual Sequential Model, _details in paper_) and some more analysis for our Anonymous Submission.

### Additional Plots/Figures (Not in the main paper, were attached as Supplementary material)
#### Bubble Plots of Dataset Vocabulary:
<table width="100%" align="center">
<tr width="100%">
  <td width="50%">
    <p align="center">
      <img src="https://user-images.githubusercontent.com/48205355/53881586-13718b80-403a-11e9-86ca-c7f7115fcc1d.png" align="center">
    </p>
<!--     <p align="center"><b>check</b></p> -->
  </td>
  <td width="50%">
    <p align="center">
      <img src="https://user-images.githubusercontent.com/48205355/53881587-13718b80-403a-11e9-9da4-8217472fffb6.png" align="center">
    </p>
<!--     <p align="center"><b>check</b></p> -->
  </td>
</tr>
  <tr> 
<td colspan="2">    
Left:  Humor, Right:  Non-Humor.  Bubble plots drawn to visualize the vocabulary of Humorous and Non-Humorous Dialogs.  Similarity between the two suggests that simple words based humor detection methods would not give good results. (Size of bubble gives measure of the frequency of words)
    </td>
  </tr>
  </table>
  
#### A tSNE plot of Visual Dialogs:

  <table>
  <tr>
    <td width="100%">
<img src="https://user-images.githubusercontent.com/48205355/53881679-474cb100-403a-11e9-9d92-c71a0c634fe9.png" align="center">
    </td>
  </tr>
  <tr>
  <td>    
A tSNE plot made by randomly selecting 1500 images (each from Humorous and Non-Humorous set) as the last frame of some visual dialog turns. Sometimes, these visual models could cheat by detecting some pattern in Humorous/Non-Humorous visual dialogs like specific camera angle etc. The above plot hints towards its absence. To visualize the plot better, each image is represented by a dot and the corresponding plot is shown below. (Current plot is slightly scaled up to ease the visibility.)
  </td>
  </tr>
    <tr>
    <td width="100%">
<img src="https://user-images.githubusercontent.com/48205355/53881680-47e54780-403a-11e9-823b-79386e56c39e.png" align="center">
    </td>
  </tr>
  <tr>
  <td>    
A green dot represents a humorous sample and red dot,  a non-humorous sample. They seem to be randomly distributed, hinting towards absence of any such bias.
  </td>
  </tr>
</table>

#### Other Dataset Statistics:
<table>
    <tr>
  <td width="33.33%">
  <img src="https://user-images.githubusercontent.com/48205355/53884884-88e15a00-4042-11e9-99a5-8fdd7a46ce68.png" align="center">
  </td>
    <td width="33.33%">
  <img src="https://user-images.githubusercontent.com/48205355/53952853-f13e4300-40f7-11e9-9444-7d9a030dc4ae.png" align="center">
  </td>
      <td width="33.33%">
  <img src="https://user-images.githubusercontent.com/48205355/53885089-ea092d80-4042-11e9-94c3-b4690723cb32.png">
  </td>
  </tr>
  <tr>
  <td width="33.33%">The figure showing average time per turn in a Dialog, across the Dataset.</td>
  <td width="33.33%">The figure showing average dialog time, across the Dataset.</td>
  <td width="33.33%">The figure showing contribution of each speaker in generating humor, across the Dataset.</td>
  </tr>
</table>

#### Baseline Models:
##### Fusion Models:
<table>
   <tr>
  <td width="50%">
  <img src="https://user-images.githubusercontent.com/48205355/53886442-0e1a3e00-4046-11e9-87a3-259d68593d62.png" align="center">
  </td>
    <td width="50%">
  <img src="https://user-images.githubusercontent.com/48205355/53886443-0e1a3e00-4046-11e9-808b-6e50ec5a9b04.png" align="center">
  </td>
  </tr>
  
  <tr>
  <td width="50%" align="center">Text based Fusion Model (TFM)</td>
  <td width="50%" align="center">Video based Fusion Model (VFM)</td>
  </tr>
</table>
  
 ##### Attention Models:
 
<table>
   <tr>
  <td width="50%">
  <img src="https://user-images.githubusercontent.com/48205355/53887017-5f76fd00-4047-11e9-97d6-7e690101f2a9.png" align="center">
  </td>
    <td width="50%">
  <img src="https://user-images.githubusercontent.com/48205355/53887018-600f9380-4047-11e9-94b9-062446d18306.png" align="center">
  </td>
  </tr>
  
  <tr>
  <td width="50%" align="center">Text based Attention Model (TAM)</td>
  <td width="50%" align="center">Video based Attention Model (VAM)</td>
  </tr>
</table>



