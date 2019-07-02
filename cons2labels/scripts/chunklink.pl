#!/usr/local/bin/perl -w

###
### Produces a convenient list of words, chunks, functions, and links from Penn TreeBank II
###
### Written by Sabine Buchholz, ILK, Tilburg University, S.Buchholz@kub.nl, 1998
###
### Modifications by Yuval Krymolowski, Bar-Ilan University, yuvalk@macs.biu.ac.il, 1999
###
### Modifications by Evgeny A. Stepanov, University of Trento, stepanov.evgeny.a@gmail.com 2014
###

sub help {
print "call as:                                                                                      \n";
print "  chunklink.pl <options> /cdrom/treebank/combined/wsj/0?/wsj_0???.mrg | more                  \n";
print "                                                                                              \n";
print "options:                                                                                      \n";
print " -s : Place a '# sentence ID' line before the word-list of each sentence                      \n";
print "      instead of at the lines of the individual words.                                        \n";
print "      The sentence ID is file/number, e.g., 0001/01.                                          \n";
print "                                                                                              \n";
print " -ns : Enumerate the words inside a sentence, instead of number in the file                   \n";
print "                                                                                              \n";
print " -B  : which sort of IOB tags to output; I tags are always inside a chunk, O tags are outside \n";
print "       possible values are: Begin (the default): B tags for word at the beginning of a chunk  \n";
print "                            End:                 E tags for word at the end of a chunk        \n";
print "                            BeginEndCombined:    B tags for word at the beginning of a chunk  \n";
print "                                                 E tags for word at the end of a chunk        \n";
print "                                                 C tags for single word chunks                \n";
print "                            Between:             B tags for words that are at the beginning   \n";
print "                                                 of a chunk and the previous chunk had the    \n";
print "                                                 same syntactic category                      \n";
print "                                    Attention! The last option applies only to the simple     \n";
print "                                    IOB tag column (e.g. 'I-NP'), not to the IOB chain column \n";
print "                                    (e.g. 'I-S/I-S/I-NP/I-NP/I-PP/B-NP'). If 'Between', the   \n";
print "                                    latter column gets the default representation 'Begin'.    \n";
print "                                                                                              \n";
print " -N  : suppress word number in output                                                         \n";
print " -p  :     ...  POS tag ...                                                                   \n";
print " -f  :          function                                                                      \n";
print " -h  :          head word                                                                     \n";
print " -H  :          head number                                                                   \n";
print " -i  :          IOB tag                                                                       \n";
print " -c  :          IOB tag chain                                                                 \n";
print " -t  :          trace information                                                             \n";
}

######################################################################
######################################################################
############################# initialize #############################
######################################################################
######################################################################

######################################################################
### load definition of object data structures to represent parse tree
######################################################################

# use nodes;

# start of nodes.pm

package terminal;
sub new {
    my $type=shift;
    my $self={};
    $self->{number}=shift;
    $self->{iob_tag}=shift;
    $self->{iob_tag_inner}=$self->{iob_tag};
    $self->{pos_tag}=shift;
    $self->{word}=shift;
    $self->{function}=shift;
    $self->{prep}='-';         # for PNP feature: not used by chunklink.pl
    $self->{adv}='-';          # for ADV feature: not used by chunklink.pl
    $self->{head_comp}='c';
    $self->{trace}=undef;
    my $obj=bless $self, $type;
    $self->{lex_head}=[$obj];  # head is reference to itself
    return $obj;
}

package non_terminal;
sub new {
    my $type=shift;
    my $self={};
    my @d=() ;
    $self->{function}=shift;
    $self->{lex_head}=undef;
    $self->{head_comp}='c';
    $self->{daughters}=shift;
    $self->{trace}=undef;    
    $self->{iob_tag}='';
    return bless $self, $type;
}

package trace;
sub new {
    my $type=shift;
    my $self={};
    $self->{function}=shift;
    $self->{kind}=shift;
    $self->{lex_head}=shift;
    $self->{reference}=shift; # trace points to filler
    $self->{head_comp}='c';
    my $obj=bless $self, $type;
    if (defined($obj->{reference})) {
	if (defined($obj->{reference}->{trace}) && ref($obj->{reference}->{trace})  eq 'ARRAY') {
	    $obj->{reference}->{trace}=[$obj,@{$obj->{reference}->{trace}}]; # filler points to trace
	}
	else {
	    $obj->{reference}->{trace}=[$obj]; # filler points to trace
	}
    }
    return $obj;
}

package main;

# return 1;

# end of nodes.pm

if (@ARGV==0) {
    help();
    exit;
}

######################################################################
### check options
######################################################################

#require "getopts.pl" ;
use Getopt::Std;
$opt_B = 'Begin' ;
$opt_n = '' ;
$opt_s = '' ;
$opt_N = 0 ; # word number
$opt_p = 0 ; # POS tag
$opt_f = 0 ; # function
$opt_h = 0 ; # head
$opt_H = 0 ; # head number
$opt_i = 0 ; # IOB tag
$opt_c = 0 ; # IOB tag chain
$opt_t = 0 ; # trace information
getopts("B:n:sNpfhHict") ;

$sent_each_word = 1 ;       # default - print the sentence for each word
if ($opt_s)
{
    $sent_each_word = 0 ;
}

$word_enumerate = 'file' ;  # default - word number runs through all the file

if ($opt_n =~ /^s/)
{
    $word_enumerate = 'sent' ; # word number runs in current sentence
}

printf("#arguments: IOB tag: $opt_B, word numbering: $word_enumerate\n") ;
printf("#columns:") ;

if ($sent_each_word)
{
  printf(" file_id sent_id") ;
}
if ($opt_N==0)
{ 
   printf(" word_id") ;
}
if ($opt_i==0)
{ 
  printf(" iob_inner") ;
}
if ($opt_p==0)
{ 
  printf(" pos") ;
}
printf(" word") ;
if ($opt_f==0)
{ 
  printf(" function") ;
}
if ($opt_h==0)
{ 
  printf(" heads") ;
}
if ($opt_H==0)
{ 
  printf(" head_ids") ;
}
if ($opt_c==0)
{ 
  printf(" iob_chain") ;
}
if ($opt_t==0)
{ 
  printf(" trace-function trace-type trace-head_ids") ;
}
printf ("\n") ;

######################################################################
### load definition of constituent head
######################################################################

head_medium();

######################################################################
### what to prune
######################################################################

### comment in if any such constituent should be pruned
$prune_always{'NAC'}=1; 
$prune_always{'QP'}=1;
$prune_always{'NX'}=1;
$prune_always{'X'}=1;

### comment in if the second constituent should be pruned 
### when appearing as daughter of the first
### in front of any potential head
$prune_if_infrontof_head{'NP'}{'ADJP'}=1;
$prune_if_infrontof_head{'NP'}{'UCP'}=1;
$prune_if_infrontof_head{'WHNP'}{'WHADJP'}=1;
$prune_if_infrontof_head{'ADJP'}{'ADVP'}=1;

### set to 1 if these should be pruned
###        0 otherwise
$prune_s_in_vp_predicative_flag=1;
$prune_s_in_vp_empty_subject_flag=1;
$prune_vp_in_vp_flag=1;
$prune_advp_in_vp_flag=1;
	   
######################################################################
######################################################################
################################ main ################################
######################################################################
######################################################################

$word_number=0;   # initialize: for unique word number over whole corpus
foreach $file (@ARGV) {                     # loop: for each treebank file
    if (defined($file)) {                   # open for reading
	open(INPUT,"<$file") or die "Cannot open $file!\n";
    }
    else {
	die "file not defined!\n";
    }

# EDIT: Evgeny A. Stepanov
# we don't want to be restriced to wsj for demo
    if ($file =~ /\/wsj_([0-9]+)\.mrg$/) {  # extract file number
	$filenumber=$1;
    }
    else {
	# die "$file does not match!\n";
	$filenumber="$file";
    }

    print STDERR "$filenumber ";
    $sentence='';                           # initialize sentence string
    $sentence_number=0;                     # initialize sentence counter
    while (defined($sentence=<INPUT>)) {    # loop: until end of file
	$result=start_read();               # read sentence into $result
	if (defined($result->{function})    # delete outermost parentheses
	    && $result->{function} eq 'NOLABEL' 
	    && @{$result->{daughters}}==1)
	    {
		$result=$result->{daughters}->[0];
	    }

# set the word counter within file or within sentence
	if ($word_enumerate eq 'file')
	{
	    $word_num = $word_number ;
	}
	else # 'sent'
	{
	    $word_num = 0 ;
	}
#print "\n################## start_read ####################\n\n"; print_parse_tree(0,$result); print "\n\n";
	prune($result);                     # prune parse tree and determine heads of constituents
#print "################## prune ####################\n\n"; print_parse_tree(0,$result); print "\n\n";
	lexicalize($result);                # lexicalize parse tree
#print "################## lexicalize ####################\n\n"; print_parse_tree(0,$result); print "\n\n";
	@flattened=flatten($result);        # flatten tree into sequence of chunks
#print "################## flatten ####################\n\n"; print_list(@flattened); print "\n\n";
	if ($opt_c==0) {                    # if {iob_tag} feature must be computed
	    cut_traces($result);            # remove traces from the tree
	    iob_chain($result);             # compute the value of the {iob_tag} feature
	}
	chunks();                           # compute IOB-tags
#print "################## chunks ####################\n\n"; print_list(@flattened); print "\n\n";
#print "################## result ####################\n\n"; 
	print_flatten();                    # print output
    }
}
print STDERR "\n$word_number words processed\n";

######################################################################
######################################################################
################################ subs ################################
######################################################################
######################################################################

######################################################################
### start_read : called by main, calls read_sentence(_,_)
######################################################################

sub start_read {
# skip all lines not of the form "   (..." (e.g. blank lines)
    while ($sentence!~/^\s*\((.*)$/ 
	   && defined($sentence=<INPUT>)) {};
    chop($sentence);
# consumes first opening bracket of sentence
# calls read_sentence to read input until corresponding closing bracket is found
    if ($sentence=~/^\s*\((.*)$/)
    { 
	$sentence=$1;
	$depth=1;
	$sentence_number++;
	$chunk_number=0; 
	undef %corefs;
	undef %tracerefs;
	return read_sentence('NOLABEL',$depth-1);
    }
    else
    {
	return \'';   # '
    }
}

######################################################################
### read_sentence : called by start_read(), calls itself, 
###                 'trace'->new(_,_,_,_), 'terminal'->new(_,_,_,_,_), 
###                 non_terminal->new(_,_)
######################################################################

sub read_sentence {
    my $label=shift(@_);             # the label of the constituent that is to be read
    my $mydepth=shift(@_);           # at which bracket depth the constituent starts/ends
    my @store=();                    # references to the daughters of the constituent will go in here

    while ($depth>$mydepth) {        # while closing bracket of constituent is not yet found
	if (length($sentence)==0) {      # if necessary:
	    if (defined($sentence=<INPUT>)) { # read in new line
		chop($sentence);
	    }
	    else {
		die "$filenumber $sentence_number ERROR (input finished before sentence was complete)!\n";
	    }
	}
	if ($sentence=~/^\(([^ ()]+) ([^ ()]+)\)(.*)$/) { # (NN man)           # terminal node
	    $tag=$1;
	    $word=$2;
	    $sentence=$3;
	    if ($tag eq '-NONE-') {                                            # null element
		if ($word=~/^(\*T\*|\*U\*|\*NOT\*|\*RNR\*|\*ICH\*|\*EXP\*|\*PPA\*|\*\?\*|\*)(.*)$/) { # trace
		    my $kind=$1;
		    my $rest=$2;
		    if (defined($rest) && $rest=~/^-([0-9]+)$/) {              # co-referenced
			if (defined($corefs{$1})) {                            # co-reference already defined (backward reference)
			    push(@store,'trace'->new('NOFUNC',$kind,undef,$corefs{$1}));
			}
			else {                                                 # co-reference not yet defined (forward reference)
			    my $ref='trace'->new('NOFUNC',$kind,undef,undef);
			    push(@store,$ref);			
			    $tracerefs{$1}=$ref;                               # store in hash for later processing
			}
		    }
		    else {                                                     # not coreferenced
			push(@store,'trace'->new('NOFUNC',$kind,undef,undef));			
		    }
		} # no else:
                  # if $tag is '-NONE-', but $word is none of the above, 
                  # it is probably the empty complementizer: (-NONE- 0)
                  # -> just ignore
	    }
	    else {                                                             # no null element
		$tag=~s/,/COMMA/g;   # replace some special characters
		$tag=~s/-LRB-/(/g;
		$tag=~s/-RRB-/)/g;
		$word=~s/,/COMMA/g;
		$word=~s/-LRB-/(/g;
		$word=~s/-RRB-/)/g;
		$word=~s/-LCB-/{/g;
		$word=~s/-RCB-/}/g;
		push(@store,'terminal'->new($word_num++,'O',$tag,$word,'NOFUNC'));
		$word_number++ ;
	    }
	}
	elsif ($sentence=~/^\((\S+)(.*)$/) { # (XP                             # beginning of non-terminal node
	    my $index;
	    my $sublabel=$1;
	    $sentence=$2;
	    $depth++;
	    if ($sublabel=~/^(.+)-([0-9]+)$/) {                                # special: co-references
		$sublabel=$1;
		$index=$2;
	    }
	    if ($sublabel=~/^(.+)=[0-9]+$/) {                                  # delete gapping-references
		$sublabel=$1;
	    }
	    my $ref=read_sentence($sublabel,$depth-1);    # read constituent
	    push(@store,$ref);                            # store constituent as daughter
	    if (defined($index)) {                                             # was coreferenced
		if (defined($tracerefs{$index})) {                             # trace had been found before filler (forward reference)
		    $tracerefs{$index}->{reference}=$ref;                      # trace points to filler
		    $ref->{trace}=[$tracerefs{$index}];                        # filler points to trace
		}
		else {                                                         # filler had been found before trace (backward reference)
		    $corefs{$index}=$ref;                                      # store in hash for later processing
		}
	    }
	}
	elsif ($sentence=~/^\((.*)$/) { # (                                    # same, without label
	    $sentence=$1;
	    $depth++;
	    push(@store,read_sentence('NOLABEL',$depth-1));
	}
	elsif ($sentence=~/^\)(.*)$/) { # )                                    # end of non-terminal node
	    $depth--;
	    $sentence=$1;
	}	
	elsif ($sentence=~/^\s+(.*)/) {                                        # blanks
	    $sentence=$1;
	}
	else {
	    print die "$filenumber $sentence_number ERROR: No match(2) $sentence!\n";
	}	
    }
    if ($depth==$mydepth) {                             # if corresponding closing bracket has been found
	return non_terminal->new($label,\@store);       # return reference to constituent
    }
    else {
	print die "$filenumber $sentence_number ERROR: $depth $mydepth\n";
    }
}

######################################################################
### prune : called by main, calls head_of(_,_)
######################################################################

sub verbs_or_adverbs_in_front {
    if (@lastnonref+@advps>0         # at least one verb or adverb in front
	&& $i==@lastnonref+@advps) { # only verbs and adverbs in front
	return 1;
    }
    else {
	return 0;
    }
}

sub simple_prune {
    splice(@$daughters,$i,1,@{$daughters->[$i]->{daughters}});        # insert contents of ADJP etc.
    $i--;
}

sub prune_drop_first {
    splice(@$daughters,$i,1,@{$s_daughters}[1..$#{$s_daughters}]); # insert all but first daughter of S
    $i--;		    
}

# prune S with (non-)empty subject inside VP
#    so that "expected to take" is ONE VP chunk
#    so that objects of predicative verbs are indeed recognized as objects
#    possibly: so that objects of controll verbs are indeed recognized as objects
sub prune_s_in_vp_condition {
    if ( $xp eq 'VP' # && $sub_labelxp eq 'S'                          # S(-...) in VP 
                      && $sub_label eq 'S'                             # S in VP: no S-CLR, S-ADV etc.
	&& verbs_or_adverbs_in_front()                                 # verbs or adverbs in front
	&& defined($daughters->[$i]->{daughters})
	&& @{$daughters->[$i]->{daughters}}>=2                         # has at least 2 daughters:
	&& $daughters->[$i]->{daughters}->[0]->{function} =~ /-SBJ/    # 1) subject: include S-NOM-SBJ
	) {
	return 1;
    }
    else {
	return 0;
    }
}
	
sub prune_s_in_vp {
    local $s_daughters=$daughters->[$i]->{daughters};
    if (prune_s_in_vp_predicative_condition()) {
	$s_daughters->[0]->{function} =~ s/-SBJ//;       # change function to "direct object": also for S-NOM-SBJ
	simple_prune();
    }
    elsif (prune_s_in_vp_empty_subject_condition()) {
	if (defined($s_daughters->[0]->{daughters}->[0]->{reference})) { # trace no longer exists
	    $s_daughters->[0]->{daughters}->[0]->{reference}->{trace}=undef;
	}
	prune_drop_first();
    }
}

#    predicative:
#    (VP (VBP make) 
#      (S 
#        (NP-SBJ (PRP them) )
#        (ADJP-PRD (JJ fearful) )))
######################################################
### can be changed so as to also apply to cases like:
#    (VP (VBP permit) 
#      (S 
#        (NP-SBJ (NN portfolio) (NNS managers) )
#        (VP (TO to) 
#          (VP (VB retain) 
sub prune_s_in_vp_predicative_condition {
    if ($prune_s_in_vp_predicative_flag
# subject may be trace if verb verb is passivized
	&& $s_daughters->[1]->{function}=~/-PRD$/                         # 2) predicative
	) {
	return 1;
    }
    else {
	return 0;
    }
}

#    empty subject and VP is infinitive or gerund:
#    (VP (VBN expected) 
#      (S 
#        (NP-SBJ (-NONE- *-1) )
#        (VP (TO to) 
#          (VP (VB take) 
#            (NP (DT another) (JJ sharp) (NN dive) )
#    (VP (VBD evaluated) 
#      (S 
#        (NP-SBJ (-NONE- *-2) )
#        (VP (VBG raising) 
#          (NP (PRP$ our) (NN bid) )))
sub prune_s_in_vp_empty_subject_condition {
    if ($prune_s_in_vp_empty_subject_flag
	&& @{$s_daughters->[0]->{daughters}}==1                        #    subject has only one daughter
	&& ref($s_daughters->[0]->{daughters}->[0]) eq 'trace'         #    which is trace
	&& $s_daughters->[1]->{function} eq 'VP'                       # 2) infinitive/gerund
	) {
	return 1;
    }
    else {
	return 0;
    }
}

sub np_condition {
    if ($xp eq 'NP'                                                  # special case of NPs
	&& $i!=@$daughters-1 
	   && defined($daughters->[$i+1]->{function})
	&& $daughters->[$i+1]->{function}=~/^NP/                      # no directly following NP
	   && defined($daughters->[$i]->{daughters})
	   && defined($daughters->[$i]->{daughters}->[-1])
	   && defined($daughters->[$i]->{daughters}->[-1]->{pos_tag})
	&& $daughters->[$i]->{daughters}->[-1]->{pos_tag} eq 'POS') { # no possessive NP
	return 1;
    }
    else {
	return 0;
    }
}

sub check_non_terminal {
    if (exists($prune_always{$sub_labelxp})) {
	simple_prune();
    }
    elsif (exists($prune_if_infrontof_head{$xp}{$sub_labelxp})
# UCP or ADJP in NP
# WHADJP in WHNP
# ADVP in ADJP
#	   && @lastnonref==0  # no possible head-word found to date: would result in "a [UCP state and local] [NP utility]"
	   && @sub_xps==0) {  # no possible head-constituent found to date: don't prune "P.V., 61 years old, ..."
	simple_prune();
    }
    elsif ($prune_vp_in_vp_flag
	   && $xp eq 'VP' && $sub_labelxp eq 'VP'    # VP in VP
	   && verbs_or_adverbs_in_front()) {      #    verbs or adverbs in front
	simple_prune();
    }
    elsif (prune_s_in_vp_condition()) {
	prune_s_in_vp();
    }
# special case of ADVP in VP: remember position of ADVP for later pruning
    elsif ($xp eq 'VP' && $sub_labelxp eq 'ADVP') { # ADVP in VP
	push(@advps,$i);
    }
# for determining the head daughter later: remember position of non-terminal daughter
    elsif ( defined($headcat{$xp}) 
	   && $sub_label=~/^($headcat{$xp})/) {
	if (not(np_condition())) {
	    push(@sub_xps,$i);                  # remember position of sub-XP
	}
    }
# for determining the head daughter later: record if coordinating conjunction found
    elsif ($sub_labelxp eq 'CONJP') { 
	$cc=1;
    }
}

sub check_terminal {
    my $postag=$daughters->[$i]->{pos_tag};
# for determining the head daughter later: remember position of terminal daughter
    if (head_of($postag,$xp)) {    # restrictions on heads of XPs
	push(@lastnonref,$i);      # remember position of last word
    }
# for determining the head daughter later: record if coordinating conjunction found
    if ($postag=~/^CC/             # coordinating conjunction
	&& $i>0) {                 # not as first word in phrase
	$cc=1;
    }
# see above: VP in VP is deleted if only verbs or adverbs in front
    elsif ($postag=~/^RB/) {       
	push(@advps,$i);
    }
}

# determine and mark head(s): default: last word or first XP is head
sub mark_head {
    if (@lastnonref>0) {                                # possible lexical head (none for NOLABEL, ...)
	$daughters->[$lastnonref[-1]]->{head_comp}='h';
    }
    elsif (@sub_xps>0) {                                # possible non-lexical head
	if ($cc==1) {                                   # coordinated structure
	    for (my $i=0; $i<@sub_xps; $i++) {
		$daughters->[$sub_xps[$i]]->{head_comp}='h';
	    }
	}
	else {                                          # no coordination
	    $daughters->[$sub_xps[0]]->{head_comp}='h';
	}
    }
}

# prune ADVP inside a VP
sub prune_advp_in_vp {
    if ($prune_advp_in_vp_flag
	&& $xp eq 'VP' && @advps>0 && @lastnonref>0) {                     # there is an ADVP in a VP
	my $i=0; 
	my $add=0; 
	my $index;
	while ($i<@advps && $advps[$i]<$lastnonref[-1]) {               # ADVP is in front of last verb
	    $index=$advps[$i]+$add;
	    if (ref($daughters->[$index]) eq 'non_terminal') {          # prune only ADVPs not pure RBs
		$add+=@{$daughters->[$index]->{daughters}}-1;           # -1: original reference is replaced
		splice(@$daughters,$index,1,@{$daughters->[$index]->{daughters}});   # insert contents of ADVP
	    }
	    $i++;
	}
    }    
}

sub prune {
    local $res=shift(@_); # the actual node
    local $i;
    local $daughters;     # the daughters of the constituent
    local $xp;            # its syntactic category
    local @sub_xps=();    # those daughters that are non-terminals
    local @lastnonref=(); #                      ... terminals
    local @advps=();      # ADVPs in VPs
    local $cc=0;          # boolean: whether coordinating conjunction was found
    local $sub_label;     # the label of the daughter (e.g. XP-FUNC)
    local $sub_labelxp;   # its syntactic category    (e.g. XP)
    if (ref($res) eq 'terminal' || ref($res) eq 'trace') { # stop at word or trace level
	return 1;
    }
    if (ref($res) ne 'non_terminal') {
	die "(prune) No match $res ".ref($res)."!\n";
    }

    if ($res->{function}=~/^([A-Z]+)/) {  # XP
	$xp=$1;                    # the syntactic category
    }
    else {
	die "(prune) No match $res->{function}=~/^([A-Z]+)/!\n";
    }

    $daughters=$res->{daughters};  # the daughters of the constituent
    for ($i=0; $i<@$daughters; $i++) {
	$sub_label=$daughters->[$i]->{function}; # the label of the daughter
	if ($sub_label=~/^([A-Z]+)/) { # XP
	    $sub_labelxp=$1;                     # its syntactic category
	}
	else {		
	    $sub_labelxp=$sub_label;
	}

	if (ref($daughters->[$i]) eq 'non_terminal') {
	    check_non_terminal();
	}
	elsif (ref($daughters->[$i]) eq 'terminal') {
	    check_terminal();
	}
	elsif (@$daughters==1) {               # trace must be only daughter
# for determining the head daughter later: remember position of trace daughter
	    push(@lastnonref,$i);      
	}
    } # end of processing the daughters: for ($i=0; $i<@$daughters; $i++) 
    
    mark_head();
    
    prune_advp_in_vp();
    
# recursive call
    for ($i=0; $i<@$daughters; $i++) {
	prune($daughters->[$i]);
    }
}


######################################################################
### head_of : called by prune(_), calls nothing
######################################################################

sub head_of {
    ($postag,$xp)=@_;
    if (defined($head{$xp})) {
	if ($postag=~/^($head{$xp})/) {
	    return 1;
	}
	else {
	    return 0;
	}
    }
    else {
	return 0;
    }
}

######################################################################
### lexicalize : called by main, calls itself
######################################################################

sub lexicalize {
    my $res=shift(@_);
    my $i;
    if (ref($res) eq 'terminal' || ref($res) eq 'trace') {  
# stop at word or trace level
	return 1;
    }
    elsif (ref($res) eq 'non_terminal') { # non_terminal
	my $daughters=$res->{daughters};
	for ($i=0; $i<@$daughters; $i++) {
	    lexicalize($daughters->[$i]);
	}
	my $headword=[];
	for ($i=0; $i<@$daughters; $i++) { # find the headword(s)
	    if ($daughters->[$i]->{head_comp} eq 'h' && defined($daughters->[$i]->{lex_head})) {
		$headword=[@$headword,@{$daughters->[$i]->{lex_head}}];
	    }
	}
	for ($i=0; $i<@$daughters; $i++) { # copy the headword(s)
	    $daughters->[$i]->{lex_head}=$headword;
	}
	$res->{lex_head}=$headword; # copy to mother
    }
    else {
	die "(lexicalize) No match $res ".ref($res)."!\n";
    }
}

######################################################################
### flatten : called by main, calls itself
######################################################################

sub flatten {
    my $res=shift(@_);
    my $i;
    
    if (ref($res) eq 'terminal')
    { # stop at word level
	return $res;
    }
    elsif (ref($res) eq 'trace')
    { # ???
	return ();
    }
    elsif (ref($res) eq 'non_terminal' && $res->{function}=~/^([A-Z]+)/)
    { # XP
	my $xp=$1;
	my $daughters=$res->{daughters};
	
	$chunk_number++;
	
	for ($i=0; $i<@$daughters; $i++)
	{ #loop on daughters
	    
	    # For terminals, initialize the IOB tag in a special field
	    if (ref($daughters->[$i]) eq 'terminal')
	    { # terminal
		$daughters->[$i]->{iob_tag_inner} =
		    "I-$xp-$chunk_number";
	    }
	    
	    # for head words
	    if ($daughters->[$i]->{head_comp} eq 'h')
	    {
		
		# find the function: inherit from above, if the upper level
		# had already a different function - do not replace it but
		# add the current function.
		
		if ($daughters->[$i]->{function}=~/^$xp/ ||
		    $daughters->[$i]->{function} eq 'NOFUNC')
		{ # NOFUNC
		    $daughters->[$i]->{function}=$res->{function}; # copy function to head(s)
		}
		else
		{
		    $daughters->[$i]->{function} =
			$daughters->[$i]->{function} .'/' . $res->{function} ;
		}
		$daughters->[$i]->{lex_head}=$res->{lex_head};
		if (ref($daughters->[$i]) ne 'trace'
		    && ref($res->{trace}) eq 'ARRAY')
		{ # copy trace information
		    if (ref($daughters->[$i]->{trace}) eq 'ARRAY') {
			$daughters->[$i]->{trace}=[@{$res->{trace}},@{$daughters->[$i]->{trace}}];
		    }
		    else {
			$daughters->[$i]->{trace}=$res->{trace};
		    }
		}
	    }
	} # end: loop on daughters
	my @output=();
	for ($i=0; $i<@$daughters; $i++) {
	    push(@output,flatten($daughters->[$i]));
	}
	return @output;
    }
    else {
	die "(flatten) No match $res ".ref($res)." $res->{function}!\n";
    }
}

######################################################################
### cut_traces : called by main, calls itself
######################################################################

sub cut_traces {
    my $res=shift(@_);
    my $i;
    
    if (ref($res) eq 'terminal' || ref($res) eq 'trace')
    { # stop at word or trace level
	return 1;
    }
    elsif (ref($res) eq 'non_terminal')
    { # XP
	my $daughters=$res->{daughters};
	for ($i=0; $i<@$daughters; $i++) {
	    cut_traces($daughters->[$i]);
	}
	for ($i=0; $i<@$daughters; $i++) {
	    if (ref($daughters->[$i]) eq 'trace'               # trace
		|| (ref($daughters->[$i]) eq 'non_terminal'    # or empty constituent
		    && @{$daughters->[$i]->{daughters}}==0)) {
		splice(@$daughters,$i,1);                     
		$i--;                                         
	    }
	}
	
    }
    else {
	die "(cut_traces) No match $res ".ref($res)."!\n";
    }
}

######################################################################
### iob_chain : called by main, calls itself
######################################################################

sub iob_chain {
    my $res=shift(@_);
    my $i;
    
    if (ref($res) eq 'terminal' || ref($res) eq 'trace')
    { # stop at word or trace level
	return 1;
    }
    elsif (ref($res) eq 'non_terminal' && $res->{function}=~/^([A-Z]+)/)
    { # XP
      my $xp=$1;
      my $daughters=$res->{daughters};

      for ($i=0; $i<@$daughters; $i++)
      { #loop on daughters

	  $daughters->[$i]->{iob_tag} = $res->{iob_tag};

        # If there is only one daughter, use the 'combined' tag
	  if (@$daughters==1) {
	      $current_iob_tag = 'C' ;
	  }
        # If we are in the first daughter, use the 'begin' tag
	  elsif ($i==0) {
	      $current_iob_tag = 'B' ;
	      $daughters->[$i]->{iob_tag} =~ s!E-!I-!g ;
	      $daughters->[$i]->{iob_tag} =~ s!C-!B-!g ;
	  }
        # If we are in the last daughter, use the 'end' tag
	  elsif ($i==@$daughters-1) {
	      $current_iob_tag = 'E' ;
	      $daughters->[$i]->{iob_tag} =~ s!B-!I-!g ;
	      $daughters->[$i]->{iob_tag} =~ s!C-!E-!g ;
	  }
	# If we are somewhere in the middle, use the 'inside' tag
	  else {
	      $current_iob_tag = 'I' ;
	      $daughters->[$i]->{iob_tag} =~ s!(B|E|C)-!I-!g ;
	  }

        # Add the IOB tag to the chain
	$daughters->[$i]->{iob_tag} .= "$current_iob_tag-$xp/" ;
      } # end: loop on daughters
      
      for ($i=0; $i<@$daughters; $i++) {
	  iob_chain($daughters->[$i]);
      }
    }
    else {
	die "(iob_chain) No match $res ".ref($res)." $res->{function}!\n";
    }
}

######################################################################
### chunks : called by main, calls nothing
######################################################################

sub chunks {
    my $old_nr=-1;
    my $head_xp='';
    my $head_nr=-1;
    my $xp;
    my $ref;
    my $prev_ref;
    my $i;
    for ($i=@flattened-1; $i>=0; $i--) { # start at end
	$ref=$flattened[$i]; # this word
	if ($i<@flattened-1) { # previous word
	    $prev_ref=$flattened[$i+1];
	}
	if ($ref->{iob_tag_inner}=~/^I-([A-Z]+)-([0-9]+)$/) { # inside
	    $xp=$1;
	    $nr=$2;
	    if ($xp=~/^WH([A-Z]+)$/) { # WHNP, WHPP, WHADVP, WHADJP
		$xp=$1;
	    }
	    if ($ref->{function}!~/^NOFUNC/) { # word is a head
		    $head_xp=$xp;
		    $head_nr=$nr;
		    $ref->{iob_tag_inner}="I-$xp"; # head is inside a chunk
#		}
	    }
	    else { # word is no head
		if ($xp eq $head_xp && $nr eq $head_nr) { # there has been a head
		    $ref->{iob_tag_inner}="I-$xp"; # word is inside a chunk
		}
		elsif ($ref->{pos_tag} eq 'POS') { # special case "'s" "'"
		    $ref->{iob_tag_inner}="I-NP"; # inside an NP
		    if (defined($prev_ref) 
			&& ($prev_ref->{iob_tag_inner} =~ /^.-NP/)) { # there is a following NP
			$nr=$old_nr; # copy number
			if (defined($flattened[$i-1])) {
			    $ref->{lex_head}=$flattened[$i-1]->{lex_head}; 
# lex_head is head of following NP (same as lex_head of previous word)
			}
		    }
		    else {
			$ref->{function}='NP'; # 's is NP of its own '
			$nr=-2;                # gets number of its own
		    }
		}
		else {
		    $ref->{iob_tag_inner}='O'; # word is outside a chunk
		}
	    }
	    if ($old_nr!=$nr) { # chunk boundary
		$ref->{iob_tag_inner} =~ s!^I-!E-!;          # end of chunk
		if (defined($prev_ref)) {
		    $prev_ref->{iob_tag_inner} =~ s!^I-!B-!; # beginning of chunk
		    $prev_ref->{iob_tag_inner} =~ s!^E-!C-!;
		}
	    }		
	    $old_nr=$nr;
	}
	else {
	    die "File $filenumber: Error: no match $ref->{iob_tag_inner}, $ref->{word}!\n";
	}
    }
    # first word in sentence must have B- or C-tag (or O-)
    $ref->{iob_tag_inner} =~ s!^I-!B-!; 
    $ref->{iob_tag_inner} =~ s!^E-!C-!;
}

######################################################################
### print_flatten : called by main, calls nothing
######################################################################

sub print_flatten {
    my $i;
    my $j;
    my $l;
    my $trace;
    my $trace_array;
    my $headword;

    if ($sent_each_word == 0)      # argument '-s' was given
    {
	# print file name and sentence number before word-list
	printf("\# Sentence %s/%02d\n", $filenumber,$sentence_number) ;
    }
    
    for ($i=0; $i<@flattened; $i++)
    { # loop on words
	if ($sent_each_word)
	{
	    printf(" %4s %2d",$filenumber,$sentence_number);
	}

	if ($opt_N==0) { 
	    printf(" %2d",$flattened[$i]->{number}); 
	}

	if ($opt_i==0) { 
	    if ($opt_B eq 'Begin') {   
		$flattened[$i]->{iob_tag_inner} =~ s!^E-!I-!g ;
		$flattened[$i]->{iob_tag_inner} =~ s!^C-!B-!g ;
	    }
	    elsif ($opt_B eq 'End') {
		$flattened[$i]->{iob_tag_inner} =~ s!^B-!I-!g ;
		$flattened[$i]->{iob_tag_inner} =~ s!^C-!E-!g ;
	    }
	    elsif ($opt_B eq 'Between') { 
		$flattened[$i]->{iob_tag_inner} =~ s!^E-!I-!g ;
		$flattened[$i]->{iob_tag_inner} =~ s!^C-!B-!g ;
		if ($flattened[$i]->{iob_tag_inner} =~ /^B-(.*)$/
		    && ($i==0                               # first word in sentence
			|| $flattened[$i-1]->{iob_tag_inner} eq 'O'
			|| substr($flattened[$i-1]->{iob_tag_inner},2) ne $1)) { # different category
		    $flattened[$i]->{iob_tag_inner}="I-$1"; # is not 'Between'
		}
	    }
	    # else: $opt_B eq 'BeginEndCombined') : nothing needs to be changed
	    printf (" %-7s", $flattened[$i]->{iob_tag_inner}) ;
	}

	if ($opt_p==0) { 
	    printf(" %-5s",$flattened[$i]->{pos_tag}); 
	}
	printf(" %-15s",$flattened[$i]->{word});
	if ($opt_f==0) { 
	    printf(" %-15s",$flattened[$i]->{function}); 
	}

	$headword=$flattened[$i]->{lex_head};
	if (defined($headword) && @$headword>0) { # has heads
	    $printed_head = $headword->[0]->{word} ;
	    if ($opt_h==0) { 
		for ($j=1; $j<@$headword; $j++) {
		    $printed_head .= '/'.$headword->[$j]->{word} ;
		}
		printf(" %-15.30s", $printed_head) ;
	    }
	    if ($opt_H==0) { 
		printf(" %3d",$headword->[0]->{number});
		for ($j=1; $j<@$headword; $j++) {
		    printf("/%d",$headword->[$j]->{number});
		}
	    }
	}
	else {
	    if ($opt_h==0) { 
		printf(" %-15s",'???');
	    }
	    if ($opt_H==0) { 
		printf(" %-10s",'???');
	    }
	}
	if ($opt_c==0) { 
	    chop $flattened[$i]->{iob_tag} ;
	    if ($opt_B eq 'Begin'     
		|| $opt_B eq 'Between') { 
# 'Between' doesn't apply to {iob_tag_inner}: take 'Begin' instead
		$flattened[$i]->{iob_tag} =~ s!E-!I-!g ;
		$flattened[$i]->{iob_tag} =~ s!C-!B-!g ;
	    }
	    elsif ($opt_B eq 'End') {
		$flattened[$i]->{iob_tag} =~ s!B-!I-!g ;
		$flattened[$i]->{iob_tag} =~ s!C-!E-!g ;
	    }
            # else: $opt_B eq 'BeginEndCombined') : nothing needs to be changed
	    printf (" %s", $flattened[$i]->{iob_tag}) ;
	}

	if (defined($flattened[$i]->{trace})) { 
	    $trace_array=$flattened[$i]->{trace};
	    for ($l=0; $l<@$trace_array; $l++) {
		$trace=$trace_array->[$l];
		if ($opt_t==0) { 
		    printf(" %10s %4s",$trace->{function},$trace->{kind});
		    $headword=$trace->{lex_head};
		    if (defined($headword) && @$headword>0) { # has heads
			printf(" %5d",$headword->[0]->{number});
			for ($j=1; $j<@$headword; $j++)
			{
			    printf("/%d",$headword->[$j]->{number});
			}
		    }
		    else {
			printf(" %5s",'???');
		    }
		}
# break circular references to allow for proper garbage collection
		undef $trace->{reference}; # delete reference from trace to filler
	    }
	    undef $flattened[$i]->{trace}; # delete references from filler to traces
	}

        print "\n";
    }
    print "\n";
}

######################################################################
### head_medium : called by main, calls nothing
######################################################################

sub head_medium {
# (ADJP-PRD (UH OK))
$head{'ADJP'}='JJ|RB|VB|IN|UH|FW|RP|\$|#|DT|NN';
# (ADVP (UH Indeed))
$head{'ADVP'}='RB|IN|TO|DT|PDT|JJ|RP|FW|LS|UH|CC|NN|CD|VB';
# (CONJP (RB as) (RB well) (IN as))
# (CONJP (RB rather) (IN than))
# (CONJP (RB Not) (RB only))
$head{'CONJP'}='CC|IN|RB'; 
# (INTJP (RB No))
$head{'INTJ'}='UH|RB|NN|VB|FW|JJ'; 
$head{'LST'}='LS|JJ|:';
$head{'NAC'}='NN';
$head{'NOLABEL'}='[A-Z]';
$head{'NP'}='NN|CD|PRP|JJ|DT|EX|IN|RB|VB|FW|SYM|UH|WP|WDT';
$head{'NX'}='NN|CD|PRP|JJ|DT|EX|FW|SYM|UH|WP|WDT';
# CC v., vs., but, plus
$head{'PP'}='IN|TO|RB|VBG|VBN|JJ|RP|CC|FW';
#$head{'PRN'}='[A-Z]';  # ',|:|-LRB-';
$head{'PRT'}='RP|IN|RB|JJ';
$head{'QP'}='CD|DT|NN|JJ';
$head{'SBAR'}='IN|WDT';
#$head{'SINV'}='VB';
#$head{'SQ'}='VB|MD';
$head{'UCP'}='JJ|NN|VB|CD';
$head{'VP'}='VB|MD|TO|JJ|NN|POS|FW|SYM';
$head{'WHADJP'}='JJ';
$head{'WHADVP'}='WRB|IN|RB|WDT';
$head{'WHNP'}='WDT|WP|CD|DT|IN|NN|JJ|RB'; # including WP$
$head{'WHPP'}='IN|TO';


$headcat{'ADJP'}='ADJP'; 
$headcat{'ADVP'}='ADVP|.*-ADV';
$headcat{'CONJP'}='CONJP';
$headcat{'FRAG'}='FRAG|INTJ|S|VP';
$headcat{'INTJ'}='S|VP|INTJ';
$headcat{'LST'}='LST';
$headcat{'NOLABEL'}='[A-Z]';
$headcat{'NP'}='NP|NX|.*-NOM';
$headcat{'NX'}='NX';
$headcat{'PP'}='PP'; 
$headcat{'PRN'}='S|VP'; # not |PRN
$headcat{'PRT'}='PRT';
$headcat{'RRC'}='S|VP'; # not |RRC
$headcat{'S'}='S$|VP|.*-PRD'; # special: not every S
$headcat{'SBAR'}='SBAR|S|WH';
$headcat{'SBARQ'}='SBARQ|SQ|WH';
$headcat{'SINV'}='SINV|VP|SBAR';
$headcat{'SQ'}='SQ|VP|S|WH';
$headcat{'UCP'}='[A-Z]+P(-[-A-Z]+)?$|S'; # not |UCP
$headcat{'VP'}='VP';
$headcat{'WHADJP'}='WHADJP|ADJP';
$headcat{'WHADVP'}='WHADVP';
$headcat{'WHNP'}='WHNP|NP';
$headcat{'WHPP'}='WHPP';
$headcat{'X'}='S|[A-Z]+P(-[-A-Z]+)?$'; # normally pruned # not |X #'
}

######################################################################
######################################################################
############################# visualize ##############################
######################################################################
######################################################################

sub print_lex_head {
    my $r=shift(@_); # reference to list of objects (terminals)
    my $i;
    if (not(defined($r))) {
	print 'undef';
    }
    elsif (@$r==0) {
	print "[]";
    }
    else {
	print "[$r->[0]->{word}";
	for ($i=1; $i<@$r; $i++)
	{
	    print ",$r->[$i]->{word}";
	}	    
	print ']';
    }
}

sub print_trace {
    my $r=shift(@_); # reference to list of objects (traces)
    my $i;
    for ($i=0; $i<@$r; $i++) {
	print "\n\ttrace->{function}=$r->[$i]->{function}, {kind}=$r->[$i]->{kind}, {lex_head}=";
	print_lex_head($r->[$i]->{lex_head});
    }	    
}

sub print_list {
    my @r=@_; # list of objects (terminals)
    my $i;
    foreach $r (@r) {
	print "terminal->{number}=$r->{number}, {iob_tag}=$r->{iob_tag}, {pos_tag}=$r->{pos_tag}, {word}=$r->{word}, {function}=$r->{function}, {head_comp}=$r->{head_comp}, {lex_head}=";
	print_lex_head($r->{lex_head});
	if (defined($r->{trace})) {
	    print ",{trace}=[";
	    print_trace($r->{trace});
	    print "\n\t]";
	}
	print "\n";
    }
}
sub print_parse_tree {
    my $s=shift(@_); # number of spaces (=indentation for output)
    my $r=shift(@_); # reference to object (parse tree)
    my $i;
    for ($i=0; $i<$s; $i++) {
	print '  ';
    }
    if (ref($r) eq 'terminal') {
	print "terminal->{number}=$r->{number}, {iob_tag}=$r->{iob_tag}, {pos_tag}=$r->{pos_tag}, {word}=$r->{word}, {function}=$r->{function}, {head_comp}=$r->{head_comp}, {lex_head}=";
	print_lex_head($r->{lex_head});
	if (defined($r->{trace})) {
	    print ",{trace}=$r->{trace}";
	}
	print "\n";
    }
    elsif (ref($r) eq 'trace') {
	print "trace->{function}=$r->{function}, {kind}=$r->{kind}, {lex_head}=";
	print_lex_head($r->{lex_head});
	print ", {reference}=";
	if (defined($r->{reference})) {
	    print "$r->{reference}";
	}
	else {
	    print 'undef';
	}
	print ", {head_comp}=$r->{head_comp}\n";
    }
    elsif (ref($r) eq 'non_terminal') {
	print "non_terminal->{function}=$r->{function}, {lex_head}=";
	print_lex_head($r->{lex_head});
	if (defined($r->{trace})) {
	    print ",{trace}=$r->{trace}";
	}
	print ", {head_comp}=$r->{head_comp}, {daughters}=[\n";
	my $daughters=$r->{daughters};
	for ($i=0; $i<@$daughters; $i++) {
	    print_parse_tree($s+1,$daughters->[$i]);
	}
	for ($i=0; $i<$s; $i++)
	{
	    print '  ';
	}
	print "]\n";
    }
}

