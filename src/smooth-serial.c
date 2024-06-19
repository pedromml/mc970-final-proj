#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#define MASK_WIDTH 15

#define COMMENT "Histogram_GPU"
#define RGB_COMPONENT_COLOR 255

typedef struct {
  int red, green, blue;
} PPMPixel;

typedef struct {
  int x, y;
  PPMPixel *data;
} PPMImage;

static PPMImage *readPPM(const char *filename) {
  char buff[16];
  PPMImage *img;
  FILE *fp;
  int c, rgb_comp_color;
  fp = fopen(filename, "rb");
  if (!fp) {
    fprintf(stderr, "Unable to open file '%s'\n", filename);
    exit(1);
  }

  if (!fgets(buff, sizeof(buff), fp)) {
    perror(filename);
    exit(1);
  }

  if (buff[0] != 'P' || buff[1] != '3') {
    fprintf(stderr, "Invalid image format (must be 'P3')\n");
    exit(1);
  }

  img = (PPMImage *)malloc(sizeof(PPMImage));
  if (!img) {
    fprintf(stderr, "Unable to allocate memory\n");
    exit(1);
  }

  c = getc(fp);
  while (c == '#') {
    while (getc(fp) != '\n')
      ;
    c = getc(fp);
  }

  ungetc(c, fp);
  if (fscanf(fp, "%d %d", &img->x, &img->y) != 2) {
    fprintf(stderr, "Invalid image size (error loading '%s')\n", filename);
    exit(1);
  }

  if (fscanf(fp, "%d", &rgb_comp_color) != 1) {
    fprintf(stderr, "Invalid rgb component (error loading '%s')\n", filename);
    exit(1);
  }

  // if (rgb_comp_color != RGB_COMPONENT_COLOR) {
  //   fprintf(stderr, "'%s' does not have 8-bits components\n", filename);
  //   exit(1);
  // }

  while (fgetc(fp) != '\n')
    ;
  img->data = (PPMPixel *)malloc(img->x * img->y * sizeof(PPMPixel));

  if (!img) {
    fprintf(stderr, "Unable to allocate memory\n");
    exit(1);
  }

  int r, g, b;
  int index = 0;
  while(fscanf(fp, "%d %d %d", &b, &g, &r) != EOF){
    PPMPixel p = {b, g, r};
    img->data[index] = p;
    index +=1;

  }

  

  // if (fread(img->data, sizeof(PPMPixel) * img->x / 2, img->y, fp) != img->y) {
  //   fprintf(stderr, "Error loading image '%s'\n", filename);
  //   exit(1);
  // }

  fclose(fp);
  return img;
}

void writePPM(PPMImage *img) {
  fprintf(stdout, "P3\n");
  // fprintf(stdout, "# %s\n", COMMENT);
  fprintf(stdout, "%d %d\n", img->x, img->y);
  fprintf(stdout, "%d\n", 1);
  // printf(sizeof(img->data) / sizeof(img->data[0]));
  // printf(sizeof(img->data));


  for(int index = 0; index < img->x * img->y; index++){

    fprintf(stdout, "%d %d %d  ", img->data[index].blue, img->data[index].green, img->data[index].red);
    if(((index + 1) % img->x == 0)){
      fprintf(stdout, "\n");
    }
  }
  // fwrite(img->data, sizeof(PPMPixel) * img->x, img->y, stdout);
  fclose(stdout);
}

void Smoothing(PPMImage *image, PPMImage *image_copy) {
  int i, j, y, x;
  int live_cells_around;
  double t;

  for (i = 0; i < image->y; i++) {
    for (j = 0; j < image->x; j++) {
      live_cells_around = 0;
      for (y = i - 1; y <= (i + 1);
           y++) {
        for (x = j - 1; x <= (j + 1);
             x++) {
          if (x >= 0 && y >= 0 && y < image->y && x < image->x) {
            live_cells_around += image_copy->data[(y * image->x) + x].red;
          } // if
        }   // for z
      }     // for y
      if(image_copy->data[(i * image->x) + j].red == 1 && live_cells_around < 2){
        image->data[(i * image->x) + j].red = 0;
        image->data[(i * image->x) + j].blue = 0;
        image->data[(i * image->x) + j].green = 0;
      }
      if(image_copy->data[(i * image->x) + j].red == 1 && (live_cells_around >= 2 && live_cells_around <= 3)){
        image->data[(i * image->x) + j].red = 1;
        image->data[(i * image->x) + j].blue = 1;
        image->data[(i * image->x) + j].green = 1;
      }
      if(image_copy->data[(i * image->x) + j].red == 0 && (live_cells_around == 3)){
        image->data[(i * image->x) + j].red = 1;
        image->data[(i * image->x) + j].blue = 1;
        image->data[(i * image->x) + j].green = 1;
      }
      if(image_copy->data[(i * image->x) + j].red == 1 && (live_cells_around > 3)){
        image->data[(i * image->x) + j].red = 0;
        image->data[(i * image->x) + j].blue = 0;
        image->data[(i * image->x) + j].green = 0;
      }
    }
  }
}

int main(int argc, char *argv[]) {
  FILE *input;
  char filename[255];
  double t;

  if (argc < 2) {
    fprintf(stderr, "Error: missing path to input file\n");
    return 1;
  }

  if ((input = fopen(argv[1], "r")) == NULL) {
    fprintf(stderr, "Error: could not open input file!\n");
    return 1;
  }

  // Read input filename
  fscanf(input, "%s\n", filename);

  // Read input file
  PPMImage *image = readPPM(filename);
  PPMImage *image_output = readPPM(filename);

  // Call Smoothing Kernel
  t = omp_get_wtime();
  Smoothing(image_output, image);
  t = omp_get_wtime() - t;

  // Write result to stdout
  writePPM(image_output);

  // Print time to stderr
  fprintf(stderr, "%lf\n", t);

  // Cleanup
  free(image);
  free(image_output);

  return 0;
}
