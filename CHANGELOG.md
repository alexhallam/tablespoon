# 0.4.5 Droping Stan

I love Stan, but I had to drop it as the backend sampler.

A lot of problems were solved as this package was developed. Multiprocessing
with cmdstanpy folder structures was a big one. As I used this package more
in production I started to dread it more than love it. Why? A lot of small
simple things. Keeping tabs on `cmdstanpy`, getting a version of python
that all of the dependancies were happy with felt tedious. In prod environments
you also have to figure out if you have C++ compilation tools like gcc. 
This is not always the case. If those tools are not available then `install_cmdstanpy`
will not work. This is additional overhead, and something I did not want to
have to worry about. Once I got
a python version 3.8>= to work well, I found that when I wanted to make 
a notebook example with Google Colab I had to wrestle with the notebook
to get a more recent version of python -- Colab comes with 3.7 by default.
The version was not all though. I had to make sure that the priority was 
reset to run the more recent version. This was not the end. I also had 
to install additional software that comes with the newer version of 
python, but does not come with the notebook -- something called `dsutils`?

Why would I care so much about the notebook environment? Well, I think 
there are a population of people who like to run code on notebooks. This
could be how they 'kick the tires' when trying a new package. If that 
experience goes well then maybe the chances of making into production is 
higher.

Though I love Stan, and it hurts to remove it, I think that these models
are simple enough to survive just fine without stans sampler. 

If you are interested in how to use stan with a python then checkout out 
version *4.4* of this package commit hash `49066323c08915e90b8b1e06a88cbddb52a726ae`

I think these changes will make the package easier for me to maintain and
very simple to integrate for production runs.
