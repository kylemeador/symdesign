#if HAVE_CONFIG_H
# include <config.h>
#endif
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include "freesasa_internal.h"
#include "pdb.h"
#include "classifier.h"
#include "coord.h"

/**
   This file contains the functions that turn lines in PDB files to
   atom records. It's all pretty messy because one needs to keep track
   of when a new chain/new residue is encountered, and skip atoms that
   are duplicates (only first of alt coordinates are used). It is both
   possible to add atoms one by one, or by reading a whole file at
   once, and the user can supply some options about what to do when
   encountering atoms that are not recognized, whether to include
   hydrogrens, hetatm, etc. The current implementation results in a
   rather convoluted logic, that is difficult to maintain, debug and
   extend.

   TODO: Refactor. 
*/
#define ATOMS_CHUNK 512
#define RESIDUES_CHUNK 64
#define CHAINS_CHUNK 64

struct atom {
    char *res_name;
    char *res_number;
    char *atom_name;
    char *symbol;
    char *line;
    int res_index;
    char chain_label;
    freesasa_atom_class the_class;
};

static const struct atom empty_atom = {
    .res_name = NULL,
    .res_number = NULL,
    .atom_name = NULL,
    .symbol = NULL,
    .line = NULL,
    .res_index = -1,
    .chain_label = '\0',
    .the_class = FREESASA_ATOM_UNKNOWN
};

struct atoms {
    int n;
    int n_alloc;
    struct atom **atom;
    double *radius;
};

struct residues {
    int n;
    int n_alloc;
    int *first_atom;
    freesasa_nodearea **reference_area;
};

struct chains {
    int n;
    int n_alloc;
    char *labels; // all chain labels found (as string)
    int *first_atom; // first atom of each chain
};

struct freesasa_structure {
    struct atoms atoms;
    struct residues residues;
    struct chains chains;
    char *classifier_name;
    coord_t *xyz;
    int model; // model number
};

static int
guess_symbol(char *symbol,
             const char *name);

static void
atom_free(struct atom *a)
{
    if (a != NULL) {
        free(a->res_name);
        free(a->res_number);
        free(a->atom_name);
        free(a->symbol);
        free(a->line);
        free(a);
    }
}

struct atoms
atoms_init()
{
    return (struct atoms) {.n = 0, .n_alloc = 0, .atom = NULL, .radius = NULL};
}

// Allocates memory in chunks, ticks up atoms->n if allocation successful
static int
atoms_alloc(struct atoms *atoms)
{
    assert(atoms);
    assert(atoms->n <= atoms->n_alloc);

    if (atoms->n == atoms->n_alloc) {
        int new_size = atoms->n_alloc + ATOMS_CHUNK;
        void *aa = atoms->atom, *ar = atoms->radius;

        atoms->atom = realloc(atoms->atom, sizeof(struct atom*) * new_size);
        if (atoms->atom == NULL) {
            atoms->atom = aa;
            return mem_fail();
        }

        for (int i = atoms->n_alloc; i < new_size; ++i) {
            atoms->atom[i] = NULL;
        }

        atoms->radius = realloc(atoms->radius, sizeof(double) * new_size);
        if (atoms->radius == NULL) {
            atoms->radius = ar;
            return mem_fail();
        }

        atoms->n_alloc = new_size;
    }
    ++atoms->n;
    return FREESASA_SUCCESS;
}

static void
atoms_dealloc(struct atoms *atoms)
{
    if (atoms) {
        struct atom **atom = atoms->atom;
        if (atom) {
            for (int i = 0; i < atoms->n; ++i)
                if (atom[i]) atom_free(atom[i]);
            free(atom);
        }
        free(atoms->radius);
        *atoms = atoms_init();
    }
}

static struct atom *
atom_new(const char *residue_name,
         const char *residue_number,
         const char *atom_name,
         const char *symbol,
         char chain_label)
{
    struct atom *a = malloc(sizeof(struct atom));
    if (a == NULL) goto memerr;

    *a = empty_atom;

    a->line = NULL;
    a->chain_label = chain_label;
    a->res_index = -1;

    a->res_name = strdup(residue_name);
    a->res_number = strdup(residue_number);
    a->atom_name = strdup(atom_name);
    a->symbol = strdup(symbol);
    a->the_class = FREESASA_ATOM_UNKNOWN;

    if (!a->res_name || !a->res_number || !a->atom_name ||
        !a->symbol) {
        goto memerr;
    }

    return a;
    
 memerr:
    mem_fail();
    atom_free(a);
    return NULL;
}

static struct atom *
atom_new_from_line(const char *line,
                   char *alt_label) 
{
    assert(line);
    const int buflen = strlen(line);
    int flag;
    struct atom *a;
    char aname[buflen], rname[buflen], rnumber[buflen], symbol[buflen];

    if (alt_label) *alt_label = freesasa_pdb_get_alt_coord_label(line);

    freesasa_pdb_get_atom_name(aname, line);
    freesasa_pdb_get_res_name(rname, line);
    freesasa_pdb_get_res_number(rnumber, line);

    flag = freesasa_pdb_get_symbol(symbol, line);
    if (flag == FREESASA_FAIL) guess_symbol(symbol,aname);

    a = atom_new(rname, rnumber, aname, symbol, freesasa_pdb_get_chain_label(line));
    
    if (a != NULL) {
        a->line = strdup(line);
        if (a->line == NULL) {
            mem_fail();
            atom_free(a);
            a = NULL;
        }
    }

    return a;
}

static struct residues
residues_init()
{
    return (struct residues)
        {.n = 0, .n_alloc = 0, .first_atom = NULL, .reference_area = NULL};
}

static int
residues_alloc(struct residues *residues)
{
    assert(residues);
    assert(residues->n <= residues->n_alloc);

    if (residues->n == residues->n_alloc) {
        int new_size = residues->n_alloc + RESIDUES_CHUNK;
        void *fa = residues->first_atom, *ra = residues->reference_area;

        residues->first_atom = realloc(residues->first_atom,
                                       sizeof(int) * new_size);
        if (residues->first_atom == NULL) {
            residues->first_atom = fa;
            return mem_fail();
        }

        residues->reference_area = realloc(residues->reference_area,
                                           sizeof(freesasa_nodearea*) * new_size);
        if (residues->reference_area == NULL) {
            residues->reference_area = ra;
            return mem_fail();
        }

        residues->n_alloc = new_size;
    }
    ++residues->n;
    return FREESASA_SUCCESS;
}

static void
residues_dealloc(struct residues *residues)
{
    if (residues) {
        free(residues->first_atom);
        if (residues->reference_area) {
            for (int i = 0; i < residues->n; ++i) {
                free(residues->reference_area[i]);
            }
        }
        free(residues->reference_area);
        *residues = residues_init();
    }
}

static struct chains
chains_init()
{
    return (struct chains) {.n = 0, .n_alloc = 0, .first_atom = NULL, .labels = NULL};
}

static int
chains_alloc(struct chains *chains)
{
    assert(chains);
    assert(chains->n <= chains->n_alloc);

    if (chains->n == chains->n_alloc) {
        int new_size = chains->n_alloc + CHAINS_CHUNK;
        void *fa = chains->first_atom, *lbl = chains->labels;

        chains->first_atom = realloc(chains->first_atom,
                                     sizeof(int) * new_size);
        if (chains->first_atom == NULL) {
            chains->first_atom = fa;
            return mem_fail();
        }

        chains->labels = realloc(chains->labels, new_size + 1);
        if (chains->labels == NULL) {
            chains->labels = lbl;
            return mem_fail();
        }

        chains->n_alloc = new_size;
    }
    ++chains->n;
    return FREESASA_SUCCESS;
}

static void
chains_dealloc(struct chains *chains)
{
    if (chains) {
        free(chains->first_atom);
        free(chains->labels);
        *chains = chains_init();
    }
}


freesasa_structure*
freesasa_structure_new(void)
{
    freesasa_structure *s = malloc(sizeof(freesasa_structure));

    if (s == NULL) goto memerr;

    s->atoms = atoms_init();
    s->residues = residues_init();
    s->chains = chains_init();
    s->xyz = freesasa_coord_new();
    s->model = 1;
    s->classifier_name = NULL;

    if (s->xyz == NULL) goto memerr;

    return s;
 memerr:
    freesasa_structure_free(s);
    mem_fail();
    return NULL;
}

void
freesasa_structure_free(freesasa_structure *s)
{
    if (s != NULL) {
        atoms_dealloc(&s->atoms);
        residues_dealloc(&s->residues);
        chains_dealloc(&s->chains);
        if (s->xyz != NULL) freesasa_coord_free(s->xyz);
        free(s->classifier_name);
        free(s);
    }
}

/**
    This function is called when either the symbol field is missing
    from an ATOM record, or when an atom is added directly using
    freesasa_structure_add_atom() or
    freesasa_structure_add_atom_wopt(). The symbol is in turn only
    used if the atom cannot be recognized by the classifier.
*/
static int
guess_symbol(char *symbol,
             const char *name) 
{
    // if the first position is empty, assume that it is a one letter element
    // e.g. " C  "
    if (name[0] == ' ') { 
        symbol[0] = ' ';
        symbol[1] = name[1];
        symbol[2] = '\0';
    } else { 
        // if the string has padding to the right, it's a
        // two-letter element, e.g. "FE  "
        if (name[3] == ' ') {
            strncpy(symbol,name,2);
            symbol[2] = '\0';
        } else { 
            // If it's a four-letter string, it's hard to say,
            // assume only the first letter signifies the element
            symbol[0] = ' ';
            symbol[1] = name[0];
            symbol[2] = '\0';
            return freesasa_warn("guessing that atom '%s' is symbol '%s'",
                                 name,symbol);
        }
    }
    return FREESASA_SUCCESS;
}
static int
structure_add_chain(freesasa_structure *s,
                    char chain_label,
                    int i_latest_atom)
{
    int n;
    if (s->chains.n == 0 || strchr(s->chains.labels, chain_label) == NULL) {

        if (chains_alloc(&s->chains) == FREESASA_FAIL)
            return fail_msg("");

        n = s->chains.n;
        s->chains.labels[n-1] = chain_label;
        s->chains.labels[n] = '\0';

        assert (strlen(s->chains.labels) == s->chains.n);

        s->chains.first_atom[n-1] = i_latest_atom;
    }
    return FREESASA_SUCCESS;
}

static int
structure_add_residue(freesasa_structure *s,
                      const freesasa_classifier *classifier,
                      const struct atom *a,
                      int i_latest_atom)
{
    int n = s->residues.n+1;
    const freesasa_nodearea *reference = NULL;

    /* register a new residue if it's the first atom, or if the
       residue number or chain label of the current atom is different
       from the previous one */
    if (!( s->residues.n == 0 ||
         (i_latest_atom > 0 &&
          (strcmp(a->res_number, s->atoms.atom[i_latest_atom-1]->res_number) ||
           a->chain_label != s->atoms.atom[i_latest_atom-1]->chain_label) ))) {
        return FREESASA_SUCCESS;
    }

    if (residues_alloc(&s->residues) == FREESASA_FAIL) {
        return fail_msg("");
    }
    s->residues.first_atom[n-1] = i_latest_atom;

    s->residues.reference_area[n-1] = NULL;
    reference = freesasa_classifier_residue_reference(classifier, a->res_name);
    if (reference != NULL) {
        s->residues.reference_area[n-1] = malloc(sizeof(freesasa_nodearea));
        if (s->residues.reference_area[n-1] == NULL)
            return mem_fail();
        *s->residues.reference_area[n-1] = *reference;
    }

    return FREESASA_SUCCESS;
}

/**
    Get the radius of an atom, and fail, warn and/or guess depending
    on the options.
 */
static int
structure_check_atom_radius(double *radius,
                            struct atom *a,
                            const freesasa_classifier* classifier,
                            int options)
{
    *radius = freesasa_classifier_radius(classifier, a->res_name, a->atom_name);
    if (*radius < 0) {
        if (options & FREESASA_HALT_AT_UNKNOWN) {
            return fail_msg("atom '%s %s' unknown",
                            a->res_name, a->atom_name);
        } else if (options & FREESASA_SKIP_UNKNOWN) {
            return freesasa_warn("skipping unknown atom '%s %s'",
                                 a->res_name, a->atom_name, a->symbol, *radius);
        } else {
            *radius = freesasa_guess_radius(a->symbol);
            if (*radius < 0) {
                *radius = +0.;
                freesasa_warn("atom '%s %s' unknown and "
                              "can't guess radius of symbol '%s', "
                              "assigning radius 0 A",
                              a->res_name, a->atom_name, a->symbol);
            } else {
                freesasa_warn("atom '%s %s' unknown, guessing element is '%s', "
                              "and radius %.3f A",
                              a->res_name, a->atom_name, a->symbol, *radius);
            }
            // do not return FREESASA_WARN here, because we will keep the atom
        }
    }
    return FREESASA_SUCCESS;
}

static int
structure_register_classifier(freesasa_structure *structure,
                              const freesasa_classifier *classifier) {
    if (structure->classifier_name == NULL) {
        structure->classifier_name = strdup(freesasa_classifier_name(classifier));
        if (structure->classifier_name == NULL) {
            return mem_fail();
        }
    } else if (strcmp(structure->classifier_name, freesasa_classifier_name(classifier)) != 0) {
        structure->classifier_name = strdup(FREESASA_CONFLICTING_CLASSIFIERS);
        if (structure->classifier_name == NULL) {
            return mem_fail();
        }
        return FREESASA_WARN;
    }
    
    return FREESASA_SUCCESS;
}

/**
   Adds an atom to the structure using the rules specified by
   'options'. If it includes FREESASA_RADIUS_FROM_* a dummy radius is
   assigned and the caller is expected to replace it with a correct
   radius later.

   The atom a should be a pointer to a heap address, this will not be cloned.
 */
static int
structure_add_atom(freesasa_structure *structure,
                   struct atom *atom,
                   double *xyz,
                   const freesasa_classifier* classifier,
                   int options)
{
    assert(structure); assert(atom); assert(xyz);
    int na, ret;
    double r;

    // let the stricter option override if both are specified
    if (options & FREESASA_SKIP_UNKNOWN && options & FREESASA_HALT_AT_UNKNOWN)
        options &= ~FREESASA_SKIP_UNKNOWN;
    
    if (classifier == NULL) {
        classifier = &freesasa_default_classifier;
    }
    structure_register_classifier(structure, classifier);

    // calculate radius and check if we should keep the atom (based on options)
    if (options & FREESASA_RADIUS_FROM_OCCUPANCY) {
        r = 1; // fix it later
    } else {
        ret = structure_check_atom_radius(&r, atom, classifier, options);
        if (ret == FREESASA_FAIL) return fail_msg("halting at unknown atom");
        if (ret == FREESASA_WARN) return FREESASA_WARN;
    }
    assert(r >= 0);

    // If it's a keeper, allocate memory
    if (atoms_alloc(&structure->atoms) == FREESASA_FAIL)
        return fail_msg("");
    na = structure->atoms.n;

    // Store coordinates
    if (freesasa_coord_append(structure->xyz, xyz, 1) == FREESASA_FAIL)
        return mem_fail();

    // Check if this is a new chain and if so add it
    if (structure_add_chain(structure, atom->chain_label, na-1) == FREESASA_FAIL)
        return mem_fail();

    // Check if this is a new residue, and if so add it
    if (structure_add_residue(structure, classifier, atom, na-1) == FREESASA_FAIL)
        return mem_fail();

    atom->the_class = freesasa_classifier_class(classifier, atom->res_name, atom->atom_name);
    atom->res_index = structure->residues.n - 1;
    structure->atoms.radius[na-1] = r;
    structure->atoms.atom[na-1] = atom;

    return FREESASA_SUCCESS;
}

/**
    Handles the reading of PDB-files, returns NULL if problems reading
    or input or malloc failure. Error-messages should explain what
    went wrong.
 */
static freesasa_structure*
from_pdb_impl(FILE *pdb_file,
              struct file_range it,
              const freesasa_classifier *classifier,
              int options)
{
    assert(pdb_file);
    size_t len = PDB_LINE_STRL;
    char *line = NULL;
    char alt, the_alt = ' ';
    double v[3], r;
    int ret;
    struct atom *a = NULL;
    freesasa_structure *s = freesasa_structure_new();
 
    if (s == NULL) return NULL;
    
    fseek(pdb_file,it.begin,SEEK_SET);
    
    while (getline(&line, &len, pdb_file) != -1 && ftell(pdb_file) <= it.end) {
        
        if (strncmp("ATOM",line,4)==0 || ( (options & FREESASA_INCLUDE_HETATM) &&
                                           (strncmp("HETATM", line, 6) == 0) )) {
            if (freesasa_pdb_ishydrogen(line) &&
                !(options & FREESASA_INCLUDE_HYDROGEN))
                continue;

            a = atom_new_from_line(line, &alt);
            if (a == NULL)
                goto cleanup;

            if ((alt != ' ' && the_alt == ' ') || (alt == ' '))
                the_alt = alt;
            else if (alt != ' ' && alt != the_alt) {
                atom_free(a);
                a = NULL;
                continue;
            }

            ret = freesasa_pdb_get_coord(v, line);
            if (ret == FREESASA_FAIL)
                goto cleanup;

            ret = structure_add_atom(s, a, v, classifier, options);
            if (ret == FREESASA_FAIL) {
                goto cleanup;
            } else if (ret == FREESASA_WARN) {
                atom_free(a);
                a = NULL;
                continue;
            }

            if (options & FREESASA_RADIUS_FROM_OCCUPANCY) {
                ret = freesasa_pdb_get_occupancy(&r, line);
                if (ret == FREESASA_FAIL)
                    goto cleanup;
                s->atoms.radius[s->atoms.n-1] = r;
            }
        }

        if (! (options & FREESASA_JOIN_MODELS)) {
            if (strncmp("MODEL",line,5)==0)  sscanf(line+10, "%d", &s->model);
            if (strncmp("ENDMDL",line,6)==0) break;
        }
    }
    
    if (s->atoms.n == 0) {
        fail_msg("input had no valid ATOM or HETATM lines");
        goto cleanup;
    }

    free(line);
    return s;

 cleanup:
    fail_msg("");
    free(line);
    atom_free(a);
    freesasa_structure_free(s);
    return NULL;
}


int
freesasa_structure_add_atom_wopt(freesasa_structure *structure,
                                 const char *atom_name,
                                 const char *residue_name,
                                 const char *residue_number,
                                 char chain_label,
                                 double x, double y, double z,
                                 const freesasa_classifier *classifier,
                                 int options)
{
    assert(structure);
    assert(atom_name); assert(residue_name); assert(residue_number);

    struct atom *a;
    char symbol[PDB_ATOM_SYMBOL_STRL+1];
    double v[3] = {x,y,z};
    int ret, warn = 0;

    // this option can not be used here, and needs to be unset
    options &= ~FREESASA_RADIUS_FROM_OCCUPANCY;

    if (guess_symbol(symbol, atom_name) == FREESASA_WARN &&
        options & FREESASA_SKIP_UNKNOWN)
        ++warn;

    a = atom_new(residue_name, residue_number, atom_name, symbol, chain_label);
    if (a == NULL) return mem_fail();

    ret = structure_add_atom(structure, a, v, classifier, options);

    if (ret == FREESASA_FAIL ||
        (ret == FREESASA_WARN && options & FREESASA_SKIP_UNKNOWN))
        atom_free(a);

    if (!ret && warn) return FREESASA_WARN;

    return ret;
}

int 
freesasa_structure_add_atom(freesasa_structure *structure,
                            const char *atom_name,
                            const char *residue_name,
                            const char *residue_number,
                            char chain_label,
                            double x, double y, double z)
{
    return freesasa_structure_add_atom_wopt(structure, atom_name, residue_name, residue_number,
                                            chain_label, x, y, z, NULL, 0);
}

freesasa_structure *
freesasa_structure_from_pdb(FILE *pdb_file,
                            const freesasa_classifier* classifier,
                            int options)
{
    assert(pdb_file);
    return from_pdb_impl(pdb_file, freesasa_whole_file(pdb_file),
                         classifier, options);
}

freesasa_structure **
freesasa_structure_array(FILE *pdb,
                         int *n,
                         const freesasa_classifier *classifier,
                         int options)
{
    assert(pdb);
    assert(n);

    struct file_range *models = NULL, *chains = NULL;
    struct file_range whole_file;
    int n_models = 0, n_chains = 0, j0, n_new_chains;
    freesasa_structure **ss = NULL, **ssb;

    if( ! (options & FREESASA_SEPARATE_MODELS ||
           options & FREESASA_SEPARATE_CHAINS) ) {
        fail_msg("options need to specify at least one of FREESASA_SEPARATE_CHAINS "
                 "and FREESASA_SEPARATE_MODELS");
        return NULL;
    }

    whole_file = freesasa_whole_file(pdb);
    n_models = freesasa_pdb_get_models(pdb,&models);

    if (n_models == FREESASA_FAIL) {
        fail_msg("problems reading PDB-file");
        return NULL;
    }
    if (n_models == 0) {
        models = &whole_file;
        n_models = 1;
    }

    //only keep first model if option not provided
    if (! (options & FREESASA_SEPARATE_MODELS) ) n_models = 1;

    //for each model read chains if requested
    if (options & FREESASA_SEPARATE_CHAINS) {
        for (int i = 0; i < n_models; ++i) {
            chains = NULL;
            n_new_chains = freesasa_pdb_get_chains(pdb, models[i], &chains, options);

            if (n_new_chains == FREESASA_FAIL) goto cleanup;
            if (n_new_chains == 0) {
                freesasa_warn("in %s(): no chains found (in model %d)", __func__, i+1);
                continue;
            }

            ssb = ss;
            ss = realloc(ss,sizeof(freesasa_structure*)*(n_chains + n_new_chains));

            if (!ss) {
                ss = ssb;
                mem_fail();
                goto cleanup;
            }

            j0 = n_chains;
            n_chains += n_new_chains;

            for (int j = 0; j < n_new_chains; ++j) ss[j0+j] = NULL;

            for (int j = 0; j < n_new_chains; ++j) {
                ss[j0+j] = from_pdb_impl(pdb, chains[j], classifier, options);
                if (ss[j0+j] == NULL) goto cleanup;
                ss[j0+j]->model = i + 1;
            }

            free(chains);
        }
        *n = n_chains;
    } else {
        ss = malloc(sizeof(freesasa_structure*)*n_models);
        if (!ss) {
            mem_fail();
            goto cleanup;
        }

        for (int i = 0; i < n_models; ++i) ss[i] = NULL;
        *n = n_models;

        for (int i = 0; i < n_models; ++i) {
            ss[i] = from_pdb_impl(pdb, models[i], classifier, options);
            if (ss[i] == NULL) goto cleanup;
            ss[i]->model = i + 1;
        }
    }

    if (*n == 0) goto cleanup;

    if (models != &whole_file) free(models);

    return ss;

 cleanup:
    if (ss) for (int i = 0; i < *n; ++i) freesasa_structure_free(ss[i]);
    if (models != &whole_file) free(models);
    free(chains);
    *n = 0;
    free(ss);
    return NULL;
}

freesasa_structure*
freesasa_structure_get_chains(const freesasa_structure *structure,
                              const char* chains)
{
    assert(structure);
    if (strlen(chains) == 0) return NULL;

    freesasa_structure *new_s = freesasa_structure_new();
    
    if (!new_s) {
        mem_fail();
        return NULL;
    }
    
    new_s->model = structure->model;

    for (int i = 0; i < structure->atoms.n; ++i) {
        struct atom *ai = structure->atoms.atom[i];
        char c = ai->chain_label;
        if (strchr(chains,c) != NULL) {
            const double *v = freesasa_coord_i(structure->xyz,i);
            int res = freesasa_structure_add_atom(new_s, ai->atom_name,
                                                  ai->res_name, ai->res_number,
                                                  c, v[0], v[1], v[2]);
            if (res == FREESASA_FAIL) {
                fail_msg("");
                goto cleanup;
            }
        }
    }

    // the following two tests could have been done by comparing the
    // chain-strings before the loop, but this logic is simpler.
    if (new_s->atoms.n == 0) {
        goto cleanup;
    }
    if (strlen(new_s->chains.labels) != strlen(chains)) {
        fail_msg("structure has chains '%s', but '%s' requested",
                 structure->chains.labels, chains);
        goto cleanup;
    }

    return new_s;

 cleanup:
    freesasa_structure_free(new_s);
    return NULL;
}

const char *
freesasa_structure_chain_labels(const freesasa_structure *structure)
{
    assert(structure);
    return structure->chains.labels;
}

const coord_t *
freesasa_structure_xyz(const freesasa_structure *structure)
{
    assert(structure);
    return structure->xyz;
}

int
freesasa_structure_n(const freesasa_structure *structure)
{
    assert(structure);
    return structure->atoms.n;
}

int
freesasa_structure_n_residues(const freesasa_structure *structure)
{
    assert(structure);
    return structure->residues.n;
}

const char *
freesasa_structure_atom_name(const freesasa_structure *structure,
                             int i)
{
    assert(structure);
    assert(i < structure->atoms.n && i >= 0);
    return structure->atoms.atom[i]->atom_name;
}

const char*
freesasa_structure_atom_res_name(const freesasa_structure *structure,
                                 int i)
{
    assert(structure);
    assert(i < structure->atoms.n && i >= 0);
    return structure->atoms.atom[i]->res_name;
}

const char*
freesasa_structure_atom_res_number(const freesasa_structure *structure,
                                   int i)
{
    assert(structure);
    assert(i < structure->atoms.n && i >= 0);
    return structure->atoms.atom[i]->res_number;
}

char
freesasa_structure_atom_chain(const freesasa_structure *structure,
                              int i)
{
    assert(structure);
    assert(i < structure->atoms.n && i >= 0);
    return structure->atoms.atom[i]->chain_label;
}
const char*
freesasa_structure_atom_symbol(const freesasa_structure *structure,
                               int i)
{
    assert(structure);
    assert(i < structure->atoms.n && i >= 0);
    return structure->atoms.atom[i]->symbol;
}

double
freesasa_structure_atom_radius(const freesasa_structure *structure,
                               int i)
{
    assert(structure);
    assert(i < structure->atoms.n && i >= 0);
    return structure->atoms.radius[i];
}

void
freesasa_structure_atom_set_radius(freesasa_structure *structure,
                                   int i,
                                   double radius)
{
    assert(structure);
    assert(i < structure->atoms.n && i >= 0);
    structure->atoms.radius[i] = radius;
}

freesasa_atom_class
freesasa_structure_atom_class(const freesasa_structure *structure,
                              int i)
{
    assert(structure);
    assert(i < structure->atoms.n && i >= 0);
    return structure->atoms.atom[i]->the_class;
}

const char *
freesasa_structure_atom_pdb_line(const freesasa_structure *structure,
                                 int i)
{
    assert(structure);
    assert(i < structure->atoms.n && i >= 0);
    return structure->atoms.atom[i]->line;
}
const freesasa_nodearea *
freesasa_structure_residue_reference(const freesasa_structure *structure,
                                     int r_i)
{
    assert(structure);
    assert(r_i >= 0 && r_i < structure->residues.n);

    return structure->residues.reference_area[r_i];

}
int
freesasa_structure_residue_atoms(const freesasa_structure *structure,
                                 int r_i,
                                 int *first,
                                 int *last)
{
    assert(structure); assert(first); assert(last);
    const int naa = structure->residues.n;
    assert(r_i >= 0 && r_i < naa);

    *first = structure->residues.first_atom[r_i];
    if (r_i == naa-1) *last = structure->atoms.n-1;
    else *last = structure->residues.first_atom[r_i+1]-1;
    assert(*last >= *first);

    return FREESASA_SUCCESS;
}

const char*
freesasa_structure_residue_name(const freesasa_structure *structure,
                                int r_i)
{
    assert(structure);
    assert(r_i < structure->residues.n && r_i >= 0);
    return structure->atoms.atom[structure->residues.first_atom[r_i]]->res_name;
}

const char*
freesasa_structure_residue_number(const freesasa_structure *structure,
                                  int r_i)
{
    assert(structure);
    assert(r_i < structure->residues.n && r_i >= 0);
    return structure->atoms.atom[structure->residues.first_atom[r_i]]->res_number;
}

char
freesasa_structure_residue_chain(const freesasa_structure *structure,
                                 int r_i)
{
    assert(structure);
    assert(r_i < structure->residues.n && r_i >= 0);

    return structure->atoms.atom[structure->residues.first_atom[r_i]]->chain_label;
}

int
freesasa_structure_n_chains(const freesasa_structure *structure)
{
    return structure->chains.n;
}

int
freesasa_structure_chain_index(const freesasa_structure *structure,
                               char chain)
{
    assert(structure);
    for (int i = 0; i < structure->chains.n; ++i) {
        if (structure->chains.labels[i] == chain) return i;
    }
    return fail_msg("chain %c not found", chain);
}

int
freesasa_structure_chain_atoms(const freesasa_structure *structure,
                               char chain,
                               int *first,
                               int *last)
{
    assert(structure);
    int c_i = freesasa_structure_chain_index(structure, chain),
        n = freesasa_structure_n_chains(structure);
    if (c_i < 0) return fail_msg("");

    *first = structure->chains.first_atom[c_i];
    if (c_i == n - 1) *last = freesasa_structure_n(structure) - 1;
    else *last = structure->chains.first_atom[c_i+1] - 1;
    assert(*last >= *first);

    return FREESASA_SUCCESS;
}

int
freesasa_structure_chain_residues(const freesasa_structure *structure,
                                  char chain,
                                  int *first,
                                  int *last)
{
   assert(structure);
   int first_atom, last_atom;
   if (freesasa_structure_chain_atoms(structure, chain, &first_atom, &last_atom))
       return fail_msg("");
   *first = structure->atoms.atom[first_atom]->res_index;
   *last = structure->atoms.atom[last_atom]->res_index;
   return FREESASA_SUCCESS;
}

const char *
freesasa_structure_classifier_name(const freesasa_structure *structure)
{
    assert(structure);
    return structure->classifier_name;
}

int
freesasa_structure_model(const freesasa_structure *structure)
{
    return structure->model;
}

const double *
freesasa_structure_coord_array(const freesasa_structure *structure)
{
    return freesasa_coord_all(structure->xyz);
}

const double *
freesasa_structure_radius(const freesasa_structure *structure)
{
    assert(structure);
    return structure->atoms.radius;
}

void
freesasa_structure_set_radius(freesasa_structure *structure,
                              const double* radii)
{
    assert(structure);
    assert(radii);
    memcpy(structure->atoms.radius, radii, structure->atoms.n*sizeof(double));
}

