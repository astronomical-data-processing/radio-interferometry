import json
from pathlib import Path
from textwrap import dedent


ROOT = Path(__file__).resolve().parents[1]


def source_lines(text: str):
    text = dedent(text).strip("\n")
    return [line + "\n" for line in text.splitlines()]


def load_notebook(relative_path: str):
    path = ROOT / relative_path
    return path, json.loads(path.read_text(encoding="utf-8"))


def save_notebook(path: Path, notebook):
    path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")


def modernize_imaging_notebook():
    path, notebook = load_notebook("5_Imaging/5_2_sampling_functions_and_psfs.ipynb")

    notebook["cells"][4]["source"] = source_lines(
        """
        import matplotlib.image as mpimg
        from pathlib import Path
        from IPython.display import Image, display, clear_output
        from astropy.io import fits
        import track_simulator
        from mpl_toolkits.mplot3d import Axes3D
        from ipywidgets import FloatSlider
        from scipy.interpolate import griddata
        from convolutional_gridder import grid_ifft
        from AA_filter import AA_filter
        import logging

        logger0 = logging.getLogger("astropy")
        logger0.setLevel(logging.CRITICAL)


        def load_fits_image(path):
            with fits.open(path) as hdul:
                data = np.asarray(hdul[0].data)
            return np.squeeze(data)


        def show_fits_image(image_or_path, title=None, cmap="viridis", vmin=None, vmax=None, ax=None):
            if isinstance(image_or_path, (str, Path)):
                image = load_fits_image(image_or_path)
            else:
                image = np.squeeze(np.asarray(image_or_path))

            if ax is None:
                _, ax = plt.subplots(figsize=(6, 5))

            im = ax.imshow(image, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
            if title is not None:
                ax.set_title(title)
            ax.set_xticks([])
            ax.set_yticks([])
            plt.colorbar(im, ax=ax)
            return image, ax
        """
    )

    simple_psf_cells = {
        73: "../data/fits/psfs/KAT-7_6h60s_dec-30_10MHz_10chans_natural-psf.fits",
        79: "../data/fits/psfs/KAT-7_0.167h60s_dec-30_10MHz_10chans_natural-psf.fits",
        84: "../data/fits/psfs/KAT-7_2h60s_dec-30_10MHz_10chans_natural-psf.fits",
        89: "../data/fits/psfs/KAT-7_12h60s_dec-30_10MHz_10chans_natural-psf.fits",
        100: "../data/fits/psfs/KAT-7_6h60s_dec-30_10MHz_100chans_natural-psf.fits",
        106: "../data/fits/psfs/KAT-7_6h60s_dec-90_10MHz_10chans_natural-psf.fits",
        112: "../data/fits/psfs/KAT-7_6h60s_dec-60_10MHz_10chans_natural-psf.fits",
        118: "../data/fits/psfs/KAT-7_6h60s_dec0_10MHz_10chans_natural-psf.fits",
        124: "../data/fits/psfs/KAT-7_6h60s_dec30_10MHz_10chans_natural-psf.fits",
    }

    for cell_index, fits_path in simple_psf_cells.items():
        notebook["cells"][cell_index]["source"] = source_lines(
            f"""
            #wsclean -name {Path(fits_path).stem.replace("_natural-psf", "")} -size 512 512 -scale 0.004 -nosmallinversion -weight natural
            # -makepsf {Path(fits_path).stem.replace("_natural-psf", "")}.ms
            show_fits_image("{fits_path}")
            """
        )

    notebook["cells"][94]["source"] = source_lines(
        """
        #wsclean -name KAT-7_6h60s_dec-30_10MHz_1chans -size 512 512 -scale 0.004 -nosmallinversion -weight natural
        # -makepsf KAT-7_6h60s_dec-30_10MHz_1chans.ms
        # 这里直接使用 astropy 读取 FITS，再用 matplotlib 显示单通道 PSF。
        show_fits_image("../data/fits/psfs/KAT-7_6h60s_dec-30_10MHz_1chans_natural-psf.fits")
        """
    )

    save_notebook(path, notebook)


def modernize_sky_models_notebook():
    path, notebook = load_notebook("6_Deconvolution/6_1_sky_models.ipynb")

    notebook["cells"][4]["source"] = source_lines(
        """
        import matplotlib.image as mpimg
        from pathlib import Path
        from IPython.display import Image
        from astropy.io import fits
        import logging

        logger0 = logging.getLogger("astropy")
        logger0.setLevel(logging.CRITICAL)


        def load_fits_image(path):
            with fits.open(path) as hdul:
                data = np.asarray(hdul[0].data)
            return np.squeeze(data)


        def show_fits_image(image_or_path, title=None, cmap="viridis", vmin=None, vmax=None, ax=None):
            if isinstance(image_or_path, (str, Path)):
                image = load_fits_image(image_or_path)
            else:
                image = np.squeeze(np.asarray(image_or_path))

            if ax is None:
                _, ax = plt.subplots(figsize=(6, 5))

            im = ax.imshow(image, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
            if title is not None:
                ax.set_title(title)
            ax.set_xticks([])
            ax.set_yticks([])
            plt.colorbar(im, ax=ax)
            return image, ax
        """
    )

    notebook["cells"][9]["source"] = source_lines(
        """
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))

        show_fits_image(
            "../data/fits/deconv/KAT-7_6h60s_dec-30_10MHz_10chans_uniform_n100-model.fits",
            title="Sky Model",
            cmap="viridis",
            vmin=-0.1,
            vmax=1.0,
            ax=axes[0],
        )
        show_fits_image(
            "../data/fits/deconv/KAT-7_6h60s_dec-30_10MHz_10chans_uniform_n100-psf.fits",
            title="KAT-7 PSF",
            cmap="viridis",
            ax=axes[1],
        )

        plt.tight_layout()
        fig.canvas.draw()
        """
    )

    notebook["cells"][12]["source"] = source_lines(
        """
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))

        skyModel = load_fits_image("../data/fits/deconv/KAT-7_6h60s_dec-30_10MHz_10chans_uniform_n100-model.fits")
        psf = load_fits_image("../data/fits/deconv/KAT-7_6h60s_dec-30_10MHz_10chans_uniform_n100-psf.fits")
        dirtyImg = load_fits_image("../data/fits/deconv/KAT-7_6h60s_dec-30_10MHz_10chans_uniform_n100-dirty.fits")
        residualImg = load_fits_image("../data/fits/deconv/KAT-7_6h60s_dec-30_10MHz_10chans_uniform_n100-residual.fits")

        # convolve the sky model with the PSF
        sampFunc = np.fft.fft2(psf)
        skyModelVis = np.fft.fft2(skyModel)
        sampModelVis = sampFunc * skyModelVis
        convImg = np.fft.fftshift(np.fft.ifft2(sampModelVis)).real + residualImg

        show_fits_image(
            convImg,
            title="PSF convolved with Sky Model",
            cmap="viridis",
            vmin=-1.0,
            vmax=3.0,
            ax=axes[0],
        )
        show_fits_image(
            dirtyImg,
            title="Dirty",
            cmap="viridis",
            vmin=-1.0,
            vmax=3.0,
            ax=axes[1],
        )
        show_fits_image(
            dirtyImg - convImg,
            title="Difference",
            cmap="viridis",
            ax=axes[2],
        )

        plt.tight_layout()
        fig.canvas.draw()
        """
    )

    notebook["cells"][22]["source"] = source_lines(
        """
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))

        show_fits_image(
            "../data/fits/deconv/KAT-7_6h60s_dec-30_10MHz_10chans_uniform_n100-residual.fits",
            title="Residual",
            cmap="viridis",
            vmin=-0.8,
            vmax=3.0,
            ax=axes[0],
        )
        show_fits_image(
            "../data/fits/deconv/KAT-7_6h60s_dec-30_10MHz_10chans_uniform_n100-image.fits",
            title="Restored",
            cmap="viridis",
            vmin=-0.8,
            vmax=3.0,
            ax=axes[1],
        )

        plt.tight_layout()
        fig.canvas.draw()
        """
    )

    save_notebook(path, notebook)


def cleanup_assignment_notebooks():
    files = [
        "6_Deconvolution/hogbom_clean.ipynb",
        "6_Deconvolution/clark_clean_assignment.ipynb",
    ]

    for relative_path in files:
        path, notebook = load_notebook(relative_path)

        source = "".join(notebook["cells"][3]["source"])
        source = source.replace("import aplpy\n", "")
        source = source.replace("#Disable astropy/aplpy logging\n", "# Disable astropy logging\n")
        source = source.replace("logger1 = logging.getLogger('aplpy')\n", "")
        source = source.replace("logger1.setLevel(logging.CRITICAL)\n", "")
        source = source.replace("logger1.setLevel(logging.CRITICAL)", "")
        notebook["cells"][3]["source"] = source_lines(source)

        for cell in notebook["cells"]:
            if cell.get("cell_type") != "code":
                continue
            joined = "".join(cell.get("source", []))
            joined = joined.replace("E:/fits/deconv/", "../data/fits/deconv/")
            joined = joined.replace("data/KAT-7_", "../data/fits/deconv/KAT-7_")
            cell["source"] = source_lines(joined)

        save_notebook(path, notebook)


def modernize_python3_legacy_notebooks():
    path, notebook = load_notebook("6_Deconvolution/clark_clean_assignment.ipynb")
    clark_source = "".join(notebook["cells"][7]["source"])
    clark_source = clark_source.replace(
        "    print '\\tMinor Cycle: fthresh: %f'%fthresh\n",
        "    print('\\tMinor Cycle: fthresh: %f' % fthresh)\n",
    )
    clark_source = clark_source.replace(
        "        #print 'iter %i, (l,m):(%i, %i), flux: %f'%(i, lmax, mmax, fmax)\n",
        "        # print('iter %i, (l,m):(%i, %i), flux: %f' % (i, lmax, mmax, fmax))\n",
    )
    notebook["cells"][7]["source"] = source_lines(clark_source)
    save_notebook(path, notebook)

    path, notebook = load_notebook("2_Mathematical_Groundwork/fft_implementation_assignment.ipynb")
    notebook["cells"][7]["source"] = source_lines(
        """
        def one_layer_FFT(x):
            \"\"\"An implementation of the 1D Cooley-Tukey FFT using one layer\"\"\"
            N = x.size
            if N % 2 > 0:
                print("Warning: length of x is not a power of two, returning DFT")
                return matrix_DFT(x)

            X_even = matrix_DFT(x[::2])
            X_odd = matrix_DFT(x[1::2])
            factor = np.exp(-2j * np.pi * np.arange(N) / N)
            half = N // 2
            return np.concatenate(
                [X_even + factor[:half] * X_odd, X_even + factor[half:] * X_odd]
            )
        """
    )
    notebook["cells"][9]["source"] = source_lines(
        """
        xTest = np.random.random(256)  # create random vector to take the DFT of

        print(np.allclose(loop_DFT(xTest), matrix_DFT(xTest)))  # returns True if all values are equal (within numerical error)
        print(np.allclose(matrix_DFT(xTest), one_layer_FFT(xTest)))  # returns True if all values are equal (within numerical error)
        """
    )
    notebook["cells"][11]["source"] = source_lines(
        """
        print("Double Loop DFT:")
        %timeit loop_DFT(xTest)
        print("\\nMatrix DFT:")
        %timeit matrix_DFT(xTest)
        print("\\nOne Layer FFT + Matrix DFT:")
        %timeit one_layer_FFT(xTest)
        """
    )
    notebook["cells"][13]["source"] = source_lines(
        """
        print(np.allclose(one_layer_FFT(xTest), np.fft.fft(xTest)))

        print("numpy FFT:")
        %timeit np.fft.fft(xTest)
        """
    )
    notebook["cells"][19]["source"] = source_lines(
        """
        print("The output of ditrad2() is correct?", np.allclose(np.fft.fft(xTest), ditrad2(xTest)))  # 2 points if true

        print("your FFT:")
        %timeit ditrad2(xTest)  # 2 point if your time < One Layer FFT + Matrix DFT
        """
    )
    notebook["cells"][25]["source"] = source_lines(
        """
        print("The output of generalFFT() is correct?", np.allclose(np.fft.fft(xTest2), generalFFT(xTest2)))  # 1 point

        print("Your generic FFT:")
        %timeit generalFFT(xTest2)  # 1 point if it runs in approximately the same time as matrix_DFT

        %timeit generalFFT(xTest3)  # 2 point if it runs faster than the xTest2 vector
        """
    )
    notebook["cells"][17]["source"] = source_lines(
        """
        def ditrad2(x):
            \"\"\"radix-2 DIT FFT
            x: list or array of N values to perform FFT on, can be real or imaginary, x must be of size 2^n
            \"\"\"
            ox = np.asarray(x, dtype="complex")  # assure the input is an array of complex values
            # INSERT: assign a value to N, the size of the FFT
            N = ...  # ??? 1 point

            if N == 1:
                return ox  # base case

            # INSERT: compute the 'even' and 'odd' components of the FFT,
            # you will recursively call ditrad() here on a subset of the input values
            # Hint: a binary tree design splits the input in half
            even = ...  # ??? 2 points
            odd = ...  # ??? 2 points

            twiddles = np.exp(-2.0j * cmath.pi * np.arange(N) / N)  # compute the twiddle factors

            # INSERT: apply the twiddle factors and return the FFT by combining the even and odd values
            # Hint: twiddle factors are only applied to the odd values
            # Hint: combing even and odd is different from the way the inputs were split apart above.
            return ...  # ??? 3 points
        """
    )
    notebook["cells"][22]["source"] = source_lines(
        """
        def generalFFT(x):
            \"\"\"radix-2 DIT FFT
            x: list or array of N values to perform FFT on, can be real or imaginary
            \"\"\"
            ox = np.asarray(x, dtype="complex")  # assure the input is an array of complex values
            # INSERT: assign a value to N, the size of the FFT
            N = ...  # ??? 1 point

            if N == 1:
                return ox  # base case
            elif ...:  # INSERT: check if the length is divisible by 2, 1 point

                # INSERT: do a FFT, use your ditrad2() code here, 3 points
                # Hint: your ditrad2() code can be copied here, and will work with only a minor modification
                return ...

            else:  # INSERT: if not divisable by 2, do a slow Fourier Transform
                return ...  # ??? 1 point
        """
    )
    save_notebook(path, notebook)


def main():
    modernize_imaging_notebook()
    modernize_sky_models_notebook()
    cleanup_assignment_notebooks()
    modernize_python3_legacy_notebooks()


if __name__ == "__main__":
    main()
